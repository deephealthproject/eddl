/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/profiling.h"

PROFILING_ENABLE_EXTERN(fpga_reshape_kernel_data_convol);

// -----------------------------------------------------------------
// fpga_reshape_kernel_data_convol
//
// This function reshapes the kernel data for the convol into the 
// proper geometry expected by the FPGA kernels. 
// An input of format O x KH x KW x I is adapted to GO x GI x CPO x CPI x KH x KW
// where I = GI * CPI and O = GO * CPO
void fpga_reshape_kernel_data_convol(ConvolDescriptor *D, int KW, int KH, int I, int O, int CPI, int CPO) {

  _debug_fpga_funcs("reshape_kernel_convol");
  PROFILING_HEADER(fpga_reshape_kernel_data_convol);

  // if I < CPI then we need to adapt to size I
  // if O < CPO then we need to adapt to size O
  int Itarget = ((I + CPI - 1) / CPI) * CPI;
  int Otarget = ((O + CPO - 1) / CPO) * CPO;
//  int Itarget = I < CPI ? CPI : I;
//  int Otarget = O < CPO ? CPO : O;

  int GI = Itarget / CPI;
  int GO = Otarget / CPO;
  // Data moved to CPU from FPGA
  fpga_copy_from_fpga(D->K, D->K->ptr, 0);
  // create a temporal buffer in cpu
  fpga_data_type *buff = (fpga_data_type *)malloc(sizeof(fpga_data_type) * Itarget * Otarget * KW * KH);
  fpga_data_type *ptrK = (fpga_data_type *)D->K->ptr;
  // reshape
  for (int i=0; i < Itarget; i++) {
    for (int o=0; o < Otarget; o++) {
      for (int kh=0; kh<KH; kh++) {
        for (int kw=0; kw<KW; kw++) {
          int gi = i / CPI;
          int cpi = i % CPI;
          int go = o / CPO;
          int cpo = o % CPO; 
          int in_addr = (o * KW * KH * I) + (i * KW * KH) + (kh * KW) + kw;
          int out_addr = (go * GI * CPO * CPI * KH * KW) + (gi * CPO * CPI * KH * KW) + (cpo * CPI * KH * KW) + (cpi * KH * KW) + (kh * KW) + kw;
          if ((i < I) && (o < O)) {
            buff[out_addr] = ptrK[in_addr];
          } else {
            buff[out_addr] = 0.f;
          }
        }
      }
    }
  }

#ifdef FPGA_DEBUG
  printf("tensor before conversion (I=%d, O=%d, Itarget=%d Otarget=%d)\n", I, O, Itarget, Otarget);
  _profile_fpga_tensor(D->K);
#endif

  if ((I < CPI) || (O < CPO)) {
    // we need to reallocate the FPGA buffer to have it larger size
    fpga_delete_tensor(D->K->fpga_device, (cl::Buffer*)D->K->fpga_ptr, D->K->fpga_tensor_id, D->K->fpga_size);
    int new_size = Itarget * Otarget * KW * KH;
    D->K->shape[0] = Otarget;
    D->K->shape[1] = Itarget;
    D->K->size = new_size;
    D->K->fpga_size = new_size;
    D->K->fpga_ptr = fpga_create_tensor(D->K->fpga_device, new_size*sizeof(float));
    D->K->fpga_consistent_buffers = 0;
    delete D->K->ptr;
    // we allocate also on cpu so to fluently emulate with cpu
    D->K->ptr = get_fmem(D->K->size,"Tensor::updateData");
    fpga_copy_to_fpga((float *)buff, D->K, 0);
  } else {
    // Write buff into tensor on the FPGA
    fpga_copy_to_fpga((float *)buff, D->K, 0);
  }

#ifdef FPGA_DEBUG
  printf("tensor after conversion\n");
  _profile_fpga_tensor(D->K);
#endif

  // remove the buffer
  free(buff);

  PROFILING_FOOTER(fpga_reshape_kernel_data_convol);
}

// -----------------------------------------------------------------
// conv2D
//
// Different implementations provided
//

// -----------------------------------------------------------------
// cpuemu version
//
// The convolution is performed on the CPU. Used when no FPGA implementation
// is provided
//
void fpga_cpuemu_conv2D(ConvolDescriptor *D) {
  // Data moved to CPU from FPGA: kernel, bias, input data
  fpga_copy_from_fpga(D->K, D->K->ptr);
  fpga_copy_from_fpga(D->bias, D->bias->ptr);
  fpga_copy_from_fpga(D->I, D->I->ptr);
  // Convolution performed on the CPU
  cpu_conv2D(D);
  // Output data sent to the FPGA from CPU
  fpga_copy_to_fpga(D->O->ptr, D->O);
  fpga_copy_memory_to_fpga(D->ptrI, (cl::Buffer *)D->fpga_ptrI, D->fpga_sizeI);
}

// --------------------------------------------------------------------------------------------
//
// fpga_conv2D
//
// main entry point for convolutions on FPGA
//
void fpga_conv2D(ConvolDescriptor *D) {
  // debug and profiling
  _debug_fpga_funcs("conv2D");
  _profile_fpga(_FPGA_CONV2D, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  int ret = 0;

  #ifdef K_ENABLED_CONV2D
  // depending on the conv parameters we select the kernel to launch
  switch (K_VERSION_CONV2D) {
    case 1 : ret = fpga_conv2D_v1X(D); return; break;
    case 2 : ret = fpga_conv2D_v2X(D); return; break;
    default: printf("Error, unsupported conv2D kernel version\n"); exit(1); break;
  }
  #endif

  if (ret == 0) {
    // we do not have any suitable Conv implementation on FPGA, then revert to CPU
    fpga_cpuemu_conv2D(D);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
}

// --------------------------------------------------------------------------------------------
//
// fpga_conv2DReLU
//
// Conv2D + ReLUA
//
void fpga_conv2DReLU(ConvolDescriptor *D)
{
  printf("function disabled for the moment\n"); exit(1);
/*  _debug_fpga_funcs("conv2DReLU");

  cl_int err;
  cl::Event event;

  // conv2D parameters
  int batch_size   = D->I->shape[0];                  // batch size
  cl::Buffer I     = *(cl::Buffer*)D->I->fpga_ptr;    // input activations
  int Irows        = D->I->shape[2];                  // rows of input image
  int Icols        = D->I->shape[3];                  // cols of input image
  int Ichannels    = D->I->shape[1];                  // input channels
  cl::Buffer K     = *(cl::Buffer*)D->K->fpga_ptr;    // kernel
  int Krows        = D->kr;                           // kernel rows
  int Kcols        = D->kc;                           // kernel cols
  cl::Buffer B     = *(cl::Buffer*)D->bias->fpga_ptr; // bias
  int use_bias     = D->use_bias;                     // whether use bias or not
  cl::Buffer O     = *(cl::Buffer*)D->O->fpga_ptr;    // output activations
  int Orows        = D->O->shape[2];                  // rows of output images
  int Ocols        = D->O->shape[3];                  // cols of output images
  int Ochannels    = D->O->shape[1];                  // output channels
  int padding_rows = D->padrt;                        // padding rows (for top and for bottom)
  int padding_cols = D->padcl;                        // padding cols (for left and right)
  int stride_rows  = D->sr;                           // rows stride
  int stride_cols  = D->sc;                           // cols stride

  // depending on the conv parameters we select the kernel to launch
  #ifdef K_ENABLED_CONV2D
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && 
      (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, 8, 8);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }

    int apply_relu = 1;
    fpga_conv2D_launch(I, Irows, Icols, Ichannels, K, B, O, Ochannels, apply_relu, k_conv2d_cpi, k_conv2d_cpo, k_conv2d_num_kernels, k_conv2d_max_rows);

    _profile_fpga_tensor(D->O);
    return;
  }
  #endif

  printf("error, Conv2DReLU cannot be run on FPGA\n");
  exit(1);*/
}

// -----------------------------------------------------------------
// conv2D_grad
//
void fpga_cpuemu_conv2D_grad(ConvolDescriptor *D) {
  fpga_copy_from_fpga(D->D, D->D->ptr);
  fpga_copy_memory_from_fpga((cl::Buffer *)D->fpga_ptrI, D->ptrI, D->fpga_sizeI);
  fpga_copy_from_fpga(D->gK, D->gK->ptr);
  fpga_copy_from_fpga(D->gbias, D->gbias->ptr);
  cpu_conv2D_grad(D);
  fpga_copy_to_fpga(D->gK->ptr, D->gK);
  fpga_copy_to_fpga(D->gbias->ptr, D->gbias);
}

void fpga_conv2D_grad(ConvolDescriptor *D)
{
  printf("fpga_conv2D_grad not implemented yet\n"); exit(1);
}

// -----------------------------------------------------------------
// conv2D_back
//
void fpga_cpuemu_conv2D_back(ConvolDescriptor *D) {
  fpga_copy_from_fpga(D->D, D->D->ptr);
  fpga_copy_memory_from_fpga((cl::Buffer *)D->fpga_ptrI, D->ptrI, D->fpga_sizeI);
  fpga_copy_from_fpga(D->K, D->K->ptr);
  cpu_conv2D_back(D);
  fpga_copy_to_fpga(D->ID->ptr, D->ID);
}

void fpga_conv2D_back(ConvolDescriptor *D)
{
  printf("fpga_conv2D_back not implemented yet\n"); exit(1);
}

#endif
