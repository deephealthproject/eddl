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
#include "eddl/hardware/fpga/fpga_hw.h"   // for buffer copies
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"  // for cpu emulation purposes
#include "eddl/profiling.h"

PROFILING_ENABLE_EXTERN(fpga_Conv2D_MAXPOOL);
PROFILING_ENABLE_EXTERN(fpga_Conv2D_RELU);
PROFILING_ENABLE_EXTERN(fpga_Conv2D_STM);
PROFILING_ENABLE_EXTERN(fpga_Conv2D_STM_ADD);
PROFILING_ENABLE_EXTERN(fpga_Conv2D_RELU_MAXPOOL);

// -----------------------------------------------------------------
// Conv2D + Maxpool
//
void fpga_conv_maxpool(ConvolDescriptor *D)
{
    // debug and profiling
  _debug_fpga_funcs("fpga_conv2D_maxpool");
  _profile_fpga(_FPGA_CONV2D_MAXPOOL, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  int ret = 0;

  int enable_relu = 0;
  int enable_stm = 0;
  int global_offset = 0;
  int enable_upper_padding = 1;
  int enable_lower_padding = 1;
  int enable_avgp = 0;
  int enable_maxp = 1;
  int enable_clipping = 0;
  int enable_shift = 0;
  int enable_add = 0;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;
  int pos_shift = 0;

  PROFILING_HEADER(fpga_Conv2D_MAXPOOL);
  ret = fpga_k_conv(D, NULL, enable_relu, enable_stm, global_offset, 
      enable_upper_padding, enable_lower_padding, enable_maxp, enable_avgp, 
      enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift);
  PROFILING_FOOTER(fpga_Conv2D_MAXPOOL);

    if (ret == 0) {
    printf("error, Conv2DMaxpool cannot be run on FPGA\n");
    exit(1);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
}

// --------------------------------------------------------------------------------------------
// Conv2D + ReLU
//
void fpga_conv_relu(ConvolDescriptor *D)
{
  _debug_fpga_funcs("fpga_conv2D_relu");
  _profile_fpga(_FPGA_CONV2D_RELU, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  int ret = 0;

  int enable_relu = 1;
  int enable_stm = 0;
  int global_offset = 0;
  int enable_upper_padding = 1;
  int enable_lower_padding = 1;
  int enable_avgp = 0;
  int enable_maxp = 0;
  int enable_clipping = 0;
  int enable_shift = 0;
  int enable_add = 0;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;
  int pos_shift = 0;

  PROFILING_HEADER(fpga_Conv2D_RELU);
  ret = fpga_k_conv(D, NULL, enable_relu, enable_stm, global_offset, 
      enable_upper_padding, enable_lower_padding, enable_maxp, enable_avgp, 
      enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift);
  PROFILING_FOOTER(fpga_Conv2D_RELU);

  if (ret == 0) {
    printf("error, Conv2DReLU cannot be run on FPGA\n");
    exit(1);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);

}

// -----------------------------------------------------------------
// Conv2D + ReLU + Maxpool
//
void fpga_conv_relu_maxpool(ConvolDescriptor *D)
{
    // debug and profiling
  _debug_fpga_funcs("fpga_conv2D_relu_maxpool");
  _profile_fpga(_FPGA_CONV2D_RELU_MAXPOOL, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  int ret = 0;

  int enable_relu = 1;
  int enable_stm = 0;
  int global_offset = 0;
  int enable_upper_padding = 1;
  int enable_lower_padding = 1;
  int enable_avgp = 0;
  int enable_maxp = 1;
  int enable_clipping = 0;
  int enable_shift = 0;
  int enable_add = 0;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;
  int pos_shift = 0;

  PROFILING_HEADER(fpga_Conv2D_RELU_MAXPOOL);
  ret = fpga_k_conv(D, NULL,enable_relu, enable_stm, global_offset, 
      enable_upper_padding, enable_lower_padding, enable_maxp, enable_avgp, 
      enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift);
  PROFILING_FOOTER(fpga_Conv2D_RELU_MAXPOOL);

  if (ret == 0) {
    printf("error, Conv2DReLUMaxpool cannot be run on FPGA\n");
    exit(1);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
}

// -----------------------------------------------------------------
// Conv2D + Softplus + tanh + mult
//
void fpga_conv_stm(ConvolDescriptor *D)
{

    // debug and profiling
  _debug_fpga_funcs("fpga_conv2D_stm");
  _profile_fpga(_FPGA_CONV2D_STM, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  int ret = 0;

  int enable_relu = 0;
  int enable_stm = 1;
  int global_offset = 0;
  int enable_upper_padding = 1;
  int enable_lower_padding = 1;
  int enable_avgp = 0;
  int enable_maxp = 0;
  int enable_clipping = 0;
  int enable_shift = 0;
  int enable_add = 0;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;
  int pos_shift = 0;

  PROFILING_HEADER(fpga_Conv2D_STM);
  ret = fpga_k_conv(D, NULL, enable_relu, enable_stm, global_offset, 
      enable_upper_padding, enable_lower_padding, enable_maxp, enable_avgp, 
      enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift);
  PROFILING_FOOTER(fpga_Conv2D_STM);

  if (ret == 0) {
    printf("error, Conv2DSTM cannot be run on FPGA\n");
    exit(1);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
}

// -----------------------------------------------------------------
// Conv2D + Softplus + tanh + mult + Add
//
void fpga_conv_stm_add(ConvolDescriptor *D, Tensor *Add)
{

    // debug and profiling
  _debug_fpga_funcs("fpga_conv2D_stm_add");
  _profile_fpga(_FPGA_CONV2D_STM_ADD, 0);
  _profile_fpga_tensor(D->I);
  _profile_fpga_tensor(D->K);
  _profile_fpga_tensor(D->bias);

  int ret = 0;

  int enable_relu = 0;
  int enable_stm = 1;
  int global_offset = 0;
  int enable_upper_padding = 1;
  int enable_lower_padding = 1;
  int enable_avgp = 0;
  int enable_maxp = 0;
  int enable_clipping = 0;
  int enable_shift = 0;
  int enable_add = 1;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;
  int pos_shift = 0;

  PROFILING_HEADER(fpga_Conv2D_STM_ADD);
  ret = fpga_k_conv(D, Add, enable_relu, enable_stm, global_offset, 
      enable_upper_padding, enable_lower_padding, enable_maxp, enable_avgp, 
      enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift);
  PROFILING_FOOTER(fpga_Conv2D_STM_ADD);

  if (ret == 0) {
    printf("error, Conv2DSTMAdd cannot be run on FPGA\n");
    exit(1);
  }

  // profiling
  _profile_fpga_tensor(D->O);
  _profile_fpga_tensor_print(D->O);
}
#endif