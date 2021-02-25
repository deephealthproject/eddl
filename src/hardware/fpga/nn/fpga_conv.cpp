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
PROFILING_ENABLE_EXTERN(fpga_Conv2D_8x8);
PROFILING_ENABLE_EXTERN(fpga_Conv2D_4x4);
PROFILING_ENABLE_EXTERN(fpga_Conv2D);

int fpga_kernel_offset(int i, int o, int kh, int kw, int I, int O, int KH, int KW) {
  return (o * KW * KH * I) + (i * KW * KH) + (kh * KW) + kw;
}

int fpga_data_offset(int i, int h, int w, int H, int W) {
  return (i * W * H) + (h * W) + w;
}

void fpga_print_data(ConvolDescriptor *D, int KW, int KH, int I, int O, int W, int H) {

  fpga_copy_from_fpga(D->K, D->K->ptr);
  fpga_copy_from_fpga(D->I, D->I->ptr);
  fpga_copy_from_fpga(D->bias, D->bias->ptr);

  float *ptr = D->K->ptr;
  // filtro 0 0
  printf("F[0][0]\n");
  float f00_00 = ptr[fpga_kernel_offset(0, 0, 0, 0, I, O, KH, KW)];
  float f00_01 = ptr[fpga_kernel_offset(0, 0, 0, 1, I, O, KH, KW)];
  float f00_02 = ptr[fpga_kernel_offset(0, 0, 0, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f00_00, f00_01, f00_02);
  float f00_10 = ptr[fpga_kernel_offset(0, 0, 1, 0, I, O, KH, KW)];
  float f00_11 = ptr[fpga_kernel_offset(0, 0, 1, 1, I, O, KH, KW)];
  float f00_12 = ptr[fpga_kernel_offset(0, 0, 1, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f00_10, f00_11, f00_12);
  float f00_20 = ptr[fpga_kernel_offset(0, 0, 2, 0, I, O, KH, KW)];
  float f00_21 = ptr[fpga_kernel_offset(0, 0, 2, 1, I, O, KH, KW)];
  float f00_22 = ptr[fpga_kernel_offset(0, 0, 2, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f00_20, f00_21, f00_22);
  // filtro 1 0
  printf("F[1][0]\n");
  float f10_00 = ptr[fpga_kernel_offset(1, 0, 0, 0, I, O, KH, KW)];
  float f10_01 = ptr[fpga_kernel_offset(1, 0, 0, 1, I, O, KH, KW)];
  float f10_02 = ptr[fpga_kernel_offset(1, 0, 0, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f10_00, f10_01, f10_02);
  float f10_10 = ptr[fpga_kernel_offset(1, 0, 1, 0, I, O, KH, KW)];
  float f10_11 = ptr[fpga_kernel_offset(1, 0, 1, 1, I, O, KH, KW)];
  float f10_12 = ptr[fpga_kernel_offset(1, 0, 1, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f10_10, f10_11, f10_12);
  float f10_20 = ptr[fpga_kernel_offset(1, 0, 2, 0, I, O, KH, KW)];
  float f10_21 = ptr[fpga_kernel_offset(1, 0, 2, 1, I, O, KH, KW)];
  float f10_22 = ptr[fpga_kernel_offset(1, 0, 2, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f10_20, f10_21, f10_22);
  // filtro 2 0
  printf("F[2][0]\n");
  float f20_00 = ptr[fpga_kernel_offset(2, 0, 0, 0, I, O, KH, KW)];
  float f20_01 = ptr[fpga_kernel_offset(2, 0, 0, 1, I, O, KH, KW)];
  float f20_02 = ptr[fpga_kernel_offset(2, 0, 0, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f20_00, f20_01, f20_02);
  float f20_10 = ptr[fpga_kernel_offset(2, 0, 1, 0, I, O, KH, KW)];
  float f20_11 = ptr[fpga_kernel_offset(2, 0, 1, 1, I, O, KH, KW)];
  float f20_12 = ptr[fpga_kernel_offset(2, 0, 1, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f20_10, f20_11, f20_12);
  float f20_20 = ptr[fpga_kernel_offset(2, 0, 2, 0, I, O, KH, KW)];
  float f20_21 = ptr[fpga_kernel_offset(2, 0, 2, 1, I, O, KH, KW)];
  float f20_22 = ptr[fpga_kernel_offset(2, 0, 2, 2, I, O, KH, KW)];
  printf("%6.4f %6.4f %6.4f\n", f20_20, f20_21, f20_22);

  ptr = D->I->ptr;
  // data 0
  printf("D[0]\n");
  float d0_00 = ptr[fpga_data_offset(0, 0, 0, H, W)];
  float d0_01 = ptr[fpga_data_offset(0, 0, 1, H, W)];
  float d0_02 = ptr[fpga_data_offset(0, 0, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d0_00, d0_01, d0_02);
  float d0_10 = ptr[fpga_data_offset(0, 1, 0, H, W)];
  float d0_11 = ptr[fpga_data_offset(0, 1, 1, H, W)];
  float d0_12 = ptr[fpga_data_offset(0, 1, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d0_10, d0_11, d0_12);
  float d0_20 = ptr[fpga_data_offset(0, 2, 0, H, W)];
  float d0_21 = ptr[fpga_data_offset(0, 2, 1, H, W)];
  float d0_22 = ptr[fpga_data_offset(0, 2, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d0_20, d0_21, d0_22);
  // data 1
  printf("D[1]\n");
  float d1_00 = ptr[fpga_data_offset(1, 0, 0, H, W)];
  float d1_01 = ptr[fpga_data_offset(1, 0, 1, H, W)];
  float d1_02 = ptr[fpga_data_offset(1, 0, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d1_00, d1_01, d1_02);
  float d1_10 = ptr[fpga_data_offset(1, 1, 0, H, W)];
  float d1_11 = ptr[fpga_data_offset(1, 1, 1, H, W)];
  float d1_12 = ptr[fpga_data_offset(1, 1, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d1_10, d1_11, d1_12);
  float d1_20 = ptr[fpga_data_offset(1, 2, 0, H, W)];
  float d1_21 = ptr[fpga_data_offset(1, 2, 1, H, W)];
  float d1_22 = ptr[fpga_data_offset(1, 2, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d1_20, d1_21, d1_22);
  // data 2
  float d2_00 = ptr[fpga_data_offset(2, 0, 0, H, W)];
  float d2_01 = ptr[fpga_data_offset(2, 0, 1, H, W)];
  float d2_02 = ptr[fpga_data_offset(2, 0, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d2_00, d2_01, d2_02);
  float d2_10 = ptr[fpga_data_offset(2, 1, 0, H, W)];
  float d2_11 = ptr[fpga_data_offset(2, 1, 1, H, W)];
  float d2_12 = ptr[fpga_data_offset(2, 1, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d2_10, d2_11, d2_12);
  float d2_20 = ptr[fpga_data_offset(2, 2, 0, H, W)];
  float d2_21 = ptr[fpga_data_offset(2, 2, 1, H, W)];
  float d2_22 = ptr[fpga_data_offset(2, 2, 2, H, W)];
  printf("%6.4f %6.4f %6.4f\n", d2_20, d2_21, d2_22);

  // bias 0
  ptr = D->bias->ptr;
  printf("BIAS[0]\n");
  float b0 = ptr[0];
  printf("%6.4f\n", b0);

  float pixel_out = (f00_11 * d0_00) + (f00_12 * d0_01) + (f00_21 * d0_10) + (f00_22 * d0_11) +
                    (f10_11 * d1_00) + (f10_12 * d1_01) + (f10_21 * d1_10) + (f10_22 * d1_11) +
                    (f20_11 * d2_00) + (f20_12 * d2_01) + (f20_21 * d2_10) + (f20_22 * d2_11) + b0;

  printf("expected pixel out: %6.4f\n", pixel_out);
}

// -----------------------------------------------------------------
// fpga_reshape_kernel_data_convol
//
// This function reshapes the kernel data for the convol into the 
// proper geometry expected by the FPGA kernels. 
// An input of format O x KH x KW x I is adapted to GO x GI x CPO x CPI x KH x KW
// where I = GI * CPI and O = GO * CPO
void fpga_reshape_kernel_data_convol(ConvolDescriptor *D, int KW, int KH, int I, int O, int CPI, int CPO) {

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
  fpga_copy_from_fpga(D->K, D->K->ptr);
  // create a temporal buffer in cpu
  float *buff = (float *)malloc(sizeof(float) * Itarget * Otarget * KW * KH);
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
            buff[out_addr] = D->K->ptr[in_addr];
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
    delete D->K->ptr;
    // we allocate also on cpu so to fluently emulate with cpu
    D->K->ptr = get_fmem(D->K->size,"Tensor::updateData");
    fpga_copy_to_fpga(buff, D->K);
  } else {
    // Write buff into tensor on the FPGA
    fpga_copy_to_fpga(buff, D->K);
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

// -------------------------------------------------------------------
// conv2D_8x8
//
// Specific convolution on FPGA with Kernel size 3x3, Padding size 1x1
// Stride size 1x1 and Batch size 1
// The kernel on the FPGA implements 4x4 convolutions (CPI=4, CPO=4)
// The output is iterated on the FPGA but the input must be iterated
// from the CPU
//
void fpga_conv2D_8x8(cl::Buffer I, int Irows, int Icols, int Ichannels, cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels, int apply_relu) {

  PROFILING_HEADER(fpga_Conv2D_8x8);

  int KW = 3;                   // kernel width
  int KH = 3;                   // kernel height
  int H = Irows;                // input channel height
  int W = Icols;                // input channel width

  // Events
  vector<cl::Event> kernel_events(1);
  // Error variable
  cl_int err;
  

  #define CPI 8
  #define CPO 8

  int I_ITER = (Ichannels + (CPI-1)) / CPI;
  int O_ITER = (Ochannels + (CPO-1)) / CPO;

  // set kernel arguments
  int arg = 0;
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, I));

  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, Irows));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, Icols));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, Irows));   // rows
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, Ichannels));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, Ochannels));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, O_ITER));

  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, apply_relu)); // relu

  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, K));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, B));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, O));
  int global_offset = 0;
  int enable_upper_padding = 1;
  int enable_lower_padding = 1;
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, global_offset));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, enable_upper_padding));
  OCL_CHECK(err, err = kernel_conv2D_8x8.setArg(arg++, enable_lower_padding));

  // Launch the Kernel
  OCL_CHECK(err, err = (*q).enqueueNDRangeKernel(kernel_conv2D_8x8, 0, 1, 1, NULL, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");

  // we wait the kernels to have completed
  OCL_CHECK(err, err = kernel_events[0].wait());

  PROFILING_FOOTER(fpga_Conv2D_8x8);

}

// -------------------------------------------------------------------
// conv2D_4x4
//
// Specific convolution on FPGA with Kernel size 3x3, Padding size 1x1
// Stride size 1x1 and Batch size 1
// The kernel on the FPGA implements 4x4 convolutions (CPI=4, CPO=4)
// The output is iterated on the FPGA but the input must be iterated
// from the CPU
//
void fpga_conv2D_4x4(cl::Buffer I, int Irows, int Icols, int Ichannels, cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels, int apply_relu) {

  PROFILING_HEADER(fpga_Conv2D_4x4);

  int KW = 3;                   // kernel width
  int KH = 3;                   // kernel height
  int H = Irows;                // input channel height
  int W = Icols;                // input channel width

  // Events
  vector<cl::Event> kernel_events(1);
  // Error variable
  cl_int err;
  

  #define CPI 4
  #define CPO 4

  int I_ITER = (Ichannels + (CPI-1)) / CPI;
  int O_ITER = (Ochannels + (CPO-1)) / CPO;

  // set kernel arguments
  int arg = 0;
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, I));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, Irows));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, Icols));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, Irows));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, Ichannels));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, Ochannels));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, O_ITER));

  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, apply_relu)); // relu

  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, K));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, B));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, O));
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, 0));   // global offset
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, 1));   // enable upper padding
  OCL_CHECK(err, err = kernel_conv2D_4x4.setArg(arg++, 1));   // enable lower padding

  // Launch the Kernel
  OCL_CHECK(err, err = (*q).enqueueNDRangeKernel(kernel_conv2D_4x4, 0, 1, 1, NULL, &kernel_events[0]));
  set_callback(kernel_events[0], "ooo_queue");
  
  // we wait the kernels to have completed
  OCL_CHECK(err, err = kernel_events[0].wait());

  PROFILING_FOOTER(fpga_Conv2D_4x4);

}


// --------------------------------------------------------------------------------------------
//
// fpga_conv2D
//
// main entry point for convolutions on FPGA
//

void fpga_conv2D(ConvolDescriptor *D)
{

  cl_int err;
  cl::Event event;

  // conv2D parameters
  int batch_size   = D->I->shape[0];     // batch size
  cl::Buffer I     = *(cl::Buffer*)D->I->fpga_ptr;    // input activations
  int Irows        = D->I->shape[2];     // rows of input image
  int Icols        = D->I->shape[3];     // cols of input image
  int Ichannels    = D->I->shape[1];     // input channels
  cl::Buffer K     = *(cl::Buffer*)D->K->fpga_ptr;    // kernel
  int Krows        = D->kr;              // kernel rows
  int Kcols        = D->kc;              // kernel cols
  cl::Buffer B     = *(cl::Buffer*)D->bias->fpga_ptr; // bias
  int use_bias     = D->use_bias;        // whether use bias or not
  cl::Buffer O     = *(cl::Buffer*)D->O->fpga_ptr;    // output activations
  int Orows        = D->O->shape[2];     // rows of output images
  int Ocols        = D->O->shape[3];     // cols of output images
  int Ochannels    = D->O->shape[1];     // output channels
  int padding_rows = D->padrt;           // padding rows (for top and for bottom)
  int padding_cols = D->padcl;           // padding cols (for left and right)
  int stride_rows  = D->sr;              // rows stride
  int stride_cols  = D->sc;              // cols stride

  _profile_fpga(_FPGA_CONV2D, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  //fpga_print_data(D, Kcols, Krows, Ichannels, Ochannels, Icols, Irows);

  // depending on the conv parameters we select the kernel to launch
  #ifdef K_ENABLED_CONV2D_8x8
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && 
      (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, 8, 8);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }
    // in case this conv performs also RELU we change the output tensor
    if (D->fpga_apply_relu) O = *(cl::Buffer*)D->fpga_relu_ptrO;
    fpga_conv2D_8x8(I, Irows, Icols, Ichannels, K, B, O, Ochannels, D->fpga_apply_relu);

    _profile_fpga_tensor(D->O);
    return;
  }
  #endif

  #ifdef K_ENABLED_CONV2D_4x4
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && 
      (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, 4, 4);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }

    // in case this conv performs also RELU we change the output tensor
    if (D->fpga_apply_relu) O = *(cl::Buffer*)D->fpga_relu_ptrO;
    fpga_conv2D_4x4(I, Irows, Icols, Ichannels, K, B, O, Ochannels, D->fpga_apply_relu);

    _profile_fpga_tensor(D->O);
    _profile_fpga_tensor_print(D->O);
    return;
  }
  #endif

  #ifdef K_ENABLED_CONV2D
  OCL_CHECK(err, err = kernel_conv2d.setArg(0, batch_size));
  OCL_CHECK(err, err = kernel_conv2d.setArg(1, I));
  OCL_CHECK(err, err = kernel_conv2d.setArg(2, Irows));    // input
  OCL_CHECK(err, err = kernel_conv2d.setArg(3, Icols));    // output
  OCL_CHECK(err, err = kernel_conv2d.setArg(4, Ichannels));
  OCL_CHECK(err, err = kernel_conv2d.setArg(5, K));
  OCL_CHECK(err, err = kernel_conv2d.setArg(6, Krows));
  OCL_CHECK(err, err = kernel_conv2d.setArg(7, Kcols));
  OCL_CHECK(err, err = kernel_conv2d.setArg(8, B));
  OCL_CHECK(err, err = kernel_conv2d.setArg(9, use_bias));
  OCL_CHECK(err, err = kernel_conv2d.setArg(10, O));
  OCL_CHECK(err, err = kernel_conv2d.setArg(11, Orows));
  OCL_CHECK(err, err = kernel_conv2d.setArg(12, Ocols));
  OCL_CHECK(err, err = kernel_conv2d.setArg(13, Ochannels));
  OCL_CHECK(err, err = kernel_conv2d.setArg(14, padding_rows));
  OCL_CHECK(err, err = kernel_conv2d.setArg(15, padding_cols));
  OCL_CHECK(err, err = kernel_conv2d.setArg(16, stride_rows));
  OCL_CHECK(err, err = kernel_conv2d.setArg(17, stride_cols));

  OCL_CHECK(err, err = (*q).enqueueTask(kernel_conv2d, NULL, &event));
  (*q=.finish();
  return;
  #endif

  // We do not have any suitable CONV implementation on FPGA, then revert to CPU
  fpga_cpuemu_conv2D(D);
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

  cl_int err;
  cl::Event event;

  // conv2D parameters
  int batch_size   = D->I->shape[0];     // batch size
  cl::Buffer I     = *(cl::Buffer*)D->I->fpga_ptr;    // input activations
  int Irows        = D->I->shape[2];     // rows of input image
  int Icols        = D->I->shape[3];     // cols of input image
  int Ichannels    = D->I->shape[1];     // input channels
  cl::Buffer K     = *(cl::Buffer*)D->K->fpga_ptr;    // kernel
  int Krows        = D->kr;              // kernel rows
  int Kcols        = D->kc;              // kernel cols
  cl::Buffer B     = *(cl::Buffer*)D->bias->fpga_ptr; // bias
  int use_bias     = D->use_bias;        // whether use bias or not
  cl::Buffer O     = *(cl::Buffer*)D->O->fpga_ptr;    // output activations
  int Orows        = D->O->shape[2];     // rows of output images
  int Ocols        = D->O->shape[3];     // cols of output images
  int Ochannels    = D->O->shape[1];     // output channels
  int padding_rows = D->padrt;           // padding rows (for top and for bottom)
  int padding_cols = D->padcl;           // padding cols (for left and right)
  int stride_rows  = D->sr;              // rows stride
  int stride_cols  = D->sc;              // cols stride

  // depending on the conv parameters we select the kernel to launch
  #ifdef K_ENABLED_CONV2D_8x8
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && 
      (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, 8, 8);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }

    fpga_conv2D_8x8(I, Irows, Icols, Ichannels, K, B, O, Ochannels, 1);

    _profile_fpga_tensor(D->O);
    return;
  }
  #endif

  #ifdef K_ENABLED_CONV2D_4x4
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && 
      (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, 4, 4);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }

    fpga_conv2D_4x4(I, Irows, Icols, Ichannels, K, B, O, Ochannels, 1);

    _profile_fpga_tensor(D->O);
    //_profile_fpga_tensor_print(D->O);
    return;
  }
  #endif

  printf("error, Conv2DReLU cannot be run on FPGA\n");
  exit(1);
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
