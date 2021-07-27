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
PROFILING_ENABLE_EXTERN(fpga_Conv2D);
// -----------------------------------------------------------------
// fpga_reshape_kernel_data_convol
//
// This function reshapes the kernel data for the convol into the 
// proper geometry expected by the FPGA kernels. 
// An input of format O x KH x KW x I is adapted to GO x GI x CPO x CPI x KH x KW
// where I = GI * CPI and O = GO * CPO
void fpga_reshape_kernel_data_convol(ConvolDescriptor *D, int KW, int KH, int I, int O, int CPI, int CPO) {

  return;
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
  //fpga_copy_memory_to_fpga(D->ptrI, (cl::Buffer *)D->fpga_ptrI, D->fpga_sizeI);
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

  int enable_relu = 0;
  int enable_stm = 0;
  float relu_factor = 0;
  int global_offset = 0;
  int enable_avgp = 0;
  int enable_maxp = 0;
  int enable_clipping = 0;
  int enable_shift = 0;
  int enable_add = 0;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;
  int pos_shift = 0;

  printf("hola\n");

  PROFILING_HEADER(fpga_Conv2D);

  ret = fpga_k_conv(D, NULL,enable_relu, enable_stm, relu_factor, global_offset, 
      enable_maxp, enable_avgp, 
      enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift);
  PROFILING_FOOTER(fpga_Conv2D);

  if (ret == 0) {
    // we do not have any suitable Conv implementation on FPGA, then revert to CPU
    fpga_cpuemu_conv2D(D);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
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
