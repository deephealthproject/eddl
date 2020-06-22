/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/nn/cpu_nn.h"

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_conv2D      = 1;
char fpga_set_cpuemu_conv2D_grad = 1;
char fpga_set_cpuemu_conv2D_back = 1;

// -----------------------------------------------------------------
// conv2D
//
//
//

void _init_channel(Tensor *T, int channel, int batch, float value) {
  int batch_size = T->shape[0];
  int channels = T->shape[1];
  int rows = T->shape[2];
  int cols = T->shape[3];
  int stride_batch = rows * cols * channels;
  int stride_channel = rows * cols;
  int stride_row = cols;

  for (int r=0; r<rows; r++) {
    for (int c=0; c<cols; c++) {
      int addr = batch * stride_batch + channel * stride_channel + r * stride_row + c;
      T->ptr[addr] = value;
    }
  }

  fpga_copy_to_fpga(T->ptr, T);
}

void _init_kernel(Tensor *K, int ichannel, int ochannel, float value) {
  int ochannels = K->shape[0];
  int ichannels = K->shape[1];
  int rows = K->shape[2];
  int cols = K->shape[3];
  int stride_ochannel = ichannels * rows * cols;
  int stride_ichannel = rows * cols;
  int stride_row = cols;

  for (int r=0; r<rows; r++) {
    for (int c=0; c<cols; c++) {
      int addr = ichannel * stride_ichannel + ochannel * stride_ochannel + r * stride_row + c;
      K->ptr[addr] = value;
    }
  }
  fpga_copy_to_fpga(K->ptr, K);
}


void _print_channel(Tensor *T, int channel, int batch) {
  int batch_size = T->shape[0];
  int channels = T->shape[1];
  int rows = T->shape[2];
  int cols = T->shape[3];
  int stride_batch = rows * cols * channels;
  int stride_channel = rows * cols;
  int stride_row = cols;

  fpga_copy_from_fpga(T, T->ptr);

  printf("channel %d, batch %d\n", channel, batch);
  for (int r=0; r<rows; r++) {
    for (int c=0; c<cols; c++) {
      int addr = batch * stride_batch + channel * stride_channel + r * stride_row + c;
      float v = T->ptr[addr];
      printf("%6.4f ", v);
    }
    printf("\n");
  }
}

void _print_kernel(Tensor *K, int ichannel, int ochannel) {
  int ochannels = K->shape[0];
  int ichannels = K->shape[1];
  int rows = K->shape[2];
  int cols = K->shape[3];
  int stride_ochannel = ichannels * rows * cols;
  int stride_ichannel = rows * cols;
  int stride_row = cols;

  fpga_copy_from_fpga(K, K->ptr);

  printf("kernel %d x %dx%d x %d\n", ichannels, rows, cols, ochannels);
  printf("ichannel %d, ochannel %d\n", ichannel, ochannel);
  for (int r=0; r<rows; r++) {
    for (int c=0; c<cols; c++) {
      int addr = ichannel * stride_ichannel + ochannel * stride_ochannel + r * stride_row + c;
      float v = K->ptr[addr];
      printf("%6.4f ", v);
    }
    printf("\n");
  } 
} 


void fpga_cpuemu_conv2D(ConvolDescriptor *D) {
  fpga_copy_from_fpga(D->K, D->K->ptr);
  fpga_copy_from_fpga(D->bias, D->bias->ptr);
  fpga_copy_from_fpga(D->I, D->I->ptr);
  cpu_conv2D(D);
  fpga_copy_to_fpga(D->O->ptr, D->O);
  fpga_copy_memory_to_fpga(D->ptrI, D->fpga_ptrI, D->fpga_sizeI);
}

void fpga_conv2D(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D, 0);
#ifndef K_ENABLED_CONV2D
  fpga_cpuemu_conv2D(D);
#else
  cl_int err;
  cl::Event event;

//  printf("conv2d\n");

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

  _init_channel(D->I, 0, 0, 1.0);
  _init_channel(D->I, 1, 0, 2.0);
  _init_channel(D->I, 2, 0, 3.0);
  _init_kernel(D->K, 0, 0, 1.0);
  _init_kernel(D->K, 1, 0, 1.0);
  _init_kernel(D->K, 2, 0, 1.0);

  _print_channel(D->I, 0, 0);
  _print_channel(D->I, 1, 0);
  _print_channel(D->I, 2, 0);

  _print_kernel(D->K, 0, 0);

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

  _print_channel(D->O, 0, 0);

 // printf("exit\n");
 //exit(1);
#endif
  _profile_fpga(_FPGA_CONV2D, 1);
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
  _profile_fpga(_FPGA_CONV2D_GRAD, 0);
  if (fpga_set_cpuemu_conv2D_grad == 1) {
    fpga_cpuemu_conv2D_grad(D);
  } else {
      printf("fpga_conv2D_grad not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_CONV2D_GRAD, 1);
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
  _profile_fpga(_FPGA_CONV2D_BACK, 0);
  if (fpga_set_cpuemu_conv2D_back == 1) {
    fpga_cpuemu_conv2D_back(D);
  } else {
      printf("fpga_conv2D_back not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_CONV2D_BACK, 1);
}
