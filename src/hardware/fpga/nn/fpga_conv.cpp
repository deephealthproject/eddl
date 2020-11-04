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

PROFILING_ENABLE_EXTERN(fpga_reshape_input_data_convol);
PROFILING_ENABLE_EXTERN(fpga_reshape_kernel_data_convol);

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
  fpga_copy_memory_to_fpga(D->ptrI, D->fpga_ptrI, D->fpga_sizeI);
}

// -------------------------------------------------------------------
// conv2D_K3x3_S1x1_P1x1_BS1_CPI4_CPO4
//
// Specific convolution on FPGA with Kernel size 3x3, Padding size 1x1
// Stride size 1x1 and Batch size 1
// The kernel on the FPGA implements 4x4 convolutions (CPI=4, CPO=4)
// The output is iterated on the FPGA but the input must be iterated
// from the CPU
//
void fpga_conv2D_K3x3_S1x1_P1x1_BS1_CPI4_CPO4(cl::Buffer I, int Irows, int Icols, int Ichannels, cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels) {

  int KW = 3;                   // kernel width
  int KH = 3;                   // kernel height
  int H = Irows;                // input channel height
  int W = Icols;                // input channel width
  int CPI = 4;                  // input channels of FPGA kernel
  int CPO = 4;                  // output channels of FPGA kernel 
  int GI = Ichannels / CPI;     // input group channels
  int GO = Ochannels / CPO;     // output group channels
  int offset_bias = 0;          // offsets within data (bias)
  int offset_kernel = 0;        // offsets within data (kernel)
  int offset_data_out = 0;      // offsets within data (output)

  // Events
  vector<cl::Event> kernel_events(GO);
  // Error variable
  cl_int err;

  // We loop the input channels
  for (int o_iter = 0; o_iter < GO; o_iter++){
    // set kernel arguments
    int arg = 0;
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, I));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, Irows));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, Icols));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, Ichannels));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, K));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, B));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, O));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, Ochannels));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, offset_bias));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, offset_kernel));
    OCL_CHECK(err, err = kernel_conv2d.setArg(arg++, offset_data_out));

    // Update the offset poiter to bias, kernels and output data
    offset_bias = offset_bias + CPO;
    offset_kernel = offset_kernel + KW * KH * CPO * GI * CPI;
    offset_data_out = offset_data_out +  H * W;

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel_conv2D_K3x3_S1x1_P1x1_BS1, 0, 1, 1, NULL, &kernel_events[o_iter]));
    set_callback(kernel_events[o_iter], "ooo_queue");
  }

  // we wait all kernels to have completed
  for (int o_iter = 0; o_iter < GO; o_iter++) {
    OCL_CHECK(err, err = kernel_events[o_iter].wait());
  }
}

// --------------------------------------------------------------------------------------------
//
// fpga_conv2D
//
// main entry point for convolutions on FPGA
//

void fpga_conv2D(ConvolDescriptor *D)
{
#if !defined(K_ENABLED_CONV2D) && !defined(K_ENABLED_CONV2D_K3X3_S1X1_P1X1_BS1)
  // We do not have any suitable CONV implementation on FPGA, then revert to CPU
  fpga_cpuemu_conv2D(D);
#else
  cl_int err;
  cl::Event event;

  // conv2D parameters
  int batch_size   = D->I->shape[0];     // batch size
  cl::Buffer I     = *D->I->fpga_ptr;    // input activations
  int Irows        = D->I->shape[2];     // rows of input image
  int Icols        = D->I->shape[3];     // cols of input image
  int Ichannels    = D->I->shape[1];     // input channels
  cl::Buffer K     = *D->K->fpga_ptr;    // kernel
  int Krows        = D->kr;              // kernel rows
  int Kcols        = D->kc;              // kernel cols
  cl::Buffer B     = *D->bias->fpga_ptr; // bias
  int use_bias     = D->use_bias;        // whether use bias or not
  cl::Buffer O     = *D->O->fpga_ptr;    // output activations
  int Orows        = D->O->shape[2];     // rows of output images
  int Ocols        = D->O->shape[3];     // cols of output images
  int Ochannels    = D->O->shape[1];     // output channels
  int padding_rows = D->padrt;           // padding rows (for top and for bottom)
  int padding_cols = D->padcl;           // padding cols (for left and right)
  int stride_rows  = D->sr;              // rows stride
  int stride_cols  = D->sc;              // cols stride

  // depending on the conv parameters we select the kernel to launch
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    fpga_conv2D_K3x3_S1x1_P1x1_BS1_CPI4_CPO4(I, Irows, Icols, Ichannels, K, B, O, Ochannels);
  } else {
    #if !defined(K_ENABLED_CONV2D)
    fpga_cpuemu_conv2D(D);
    #else
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

    OCL_CHECK(err, err = q.enqueueTask(kernel_conv2d, NULL, &event));
    q.finish();
    #endif
  }
#endif
}

// -----------------------------------------------------------------
// fpga_reshape_input_data_convol
//
// This function reshapes the input data for the convol into the 
// proper geometry expected by the FPGA kernels. 
// An input of format I x H x W is adapted to GI x H x W x CPI 
// where I = GI * CPI
void fpga_reshape_input_data_convol(ConvolDescriptor *D, int I, int H, int W, int CPI) {

  PROFILING_HEADER(fpga_reshape_input_data_convol);

  // Data moved to CPU from FPGA
  fpga_copy_from_fpga(D->I, D->I->ptr);
  // create a temporal buffer in cpu
  float *buff = (float *)malloc(sizeof(float) * I * W * H);
  // reshape
  for (int i=0; i < I; i++) {
    for (int h=0; h<H; h++) {
      for (int w=0; w<W; w++) {
        int in_addr = (i * H * W) + (h * W) + w;
        int gi = i / CPI;
        int ii = i % CPI;
        int out_addr = (gi * CPI * H * W) + (h * W * CPI) + (w * CPI) + ii;
        buff[out_addr] = D->I->ptr[in_addr];
      }
    }
  }
  // Write buff into tensor on the FPGA
  fpga_copy_to_fpga(buff, D->I);

  // remove the buffer
  free(buff);

  PROFILING_FOOTER(fpga_reshape_input_data_convol);
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

  int GI = I / CPI;
  int GO = O / CPO;
  // Data moved to CPU from FPGA
  fpga_copy_from_fpga(D->K, D->K->ptr);
  // create a temporal buffer in cpu
  float *buff = (float *)malloc(sizeof(float) * I * O * KW * KH);
  // reshape
  for (int i=0; i < I; i++) {
    for (int o=0; o < O; o++) {
      for (int kh=0; kh<KH; kh++) {
        for (int kw=0; kw<KW; kw++) {
          int gi = i / CPI;
          int cpi = i % CPI;
          int go = o / CPO;
          int cpo = o % CPO; 
          int in_addr = (o * KW * KH * I) + (kh * KW * I) + (kw * I) + i;
          int out_addr = (go * GI * CPO * CPI * KH * KW) + (gi * CPO * CPI * KH * KW) + (cpo * CPI * KH * KW) + (cpi * KH * KW) + (kh * KW) + kw;
          buff[out_addr] = D->K->ptr[in_addr];
        }
      }
    }
  }
  // Write buff into tensor on the FPGA
  fpga_copy_to_fpga(buff, D->K);

  // remove the buffer
  free(buff);

  PROFILING_FOOTER(fpga_reshape_kernel_data_convol);
}

// -----------------------------------------------------------------
// conv2D_grad
//
void fpga_cpuemu_conv2D_grad(ConvolDescriptor *D) {
  fpga_copy_from_fpga(D->D, D->D->ptr);
  fpga_copy_memory_from_fpga(D->fpga_ptrI, D->ptrI, D->fpga_sizeI);
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
  fpga_copy_memory_from_fpga(D->fpga_ptrI, D->ptrI, D->fpga_sizeI);
  fpga_copy_from_fpga(D->K, D->K->ptr);
  cpu_conv2D_back(D);
  fpga_copy_to_fpga(D->ID->ptr, D->ID);
}

void fpga_conv2D_back(ConvolDescriptor *D)
{
  printf("fpga_conv2D_back not implemented yet\n"); exit(1);
}

#endif
