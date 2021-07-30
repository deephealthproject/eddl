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
#include "eddl/profiling.h"

extern vector<cl::Event> kernel_events; // Kernel events (completion)

PROFILING_ENABLE_EXTERN(fpga_hlsinf);

// -----------------------------------------------------------------
// HLSinf 
//
//
//

// HLSinf_kernel: This function launches the kernel execution on the FPGA
void HLSinf_kernel(cl::Buffer I, cl::Buffer I_add, int H, int W, int rows, int PT, int PB, int PL, int PR, int SH, int SW,
                      int Ichannels, int Ochannels,  int first_o_iter, int last_o_iter, int enable_relu, int enable_stm, float relu_factor,
                      cl::Buffer K, cl::Buffer B, cl::Buffer O, int global_offset, int enable_maxp, int enable_avgp,
                      int enable_clipping, int enable_shift, int enable_add, int min_clip, int max_clip,
                      int dir_shift, int pos_shift, int CPI, int CPO, int kernel_id) {

  // Error variable
  cl_int err;

  // set kernel arguments
  int arg = 0;

  // input iterations
  int I_ITER = (Ichannels + (CPI-1)) / CPI;

#ifdef DEBUG_VERBOSE
  printf("I_ITER %d, first %d last %d enable_relu %d relu_factor %f enable_stm %d enable_maxp %d enable_avgp %d enable_clip %d enable_shift %d enable_add %d\n PT %d PB %d PL %d PR %d SH %d SW %d\n",
                  I_ITER, first_o_iter, last_o_iter, enable_relu, relu_factor, enable_stm, enable_maxp, enable_avgp, enable_clipping, enable_shift,
                  enable_add, PT, PB, PL, PR, SH, SW);
  printf("H %d W %d rows %d Ich %d Och %d\n", H, W, rows, Ichannels, Ochannels);
#endif

  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I_add));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, H));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, W));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, rows));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, PT));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, PB));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, PL));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, PR));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, SH));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, SW));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Ichannels));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Ochannels));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, first_o_iter));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, last_o_iter));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_relu));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_stm));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, relu_factor));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, K));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, B));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, O));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, global_offset));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_maxp));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_avgp));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_clipping));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_shift));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_add));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, min_clip));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, max_clip));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, dir_shift));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, pos_shift));

  // Launch the Kernel
  OCL_CHECK(err, err = (*q).enqueueNDRangeKernel(kernel_conv2D[kernel_id], 0, 1, 1, NULL, &kernel_events[kernel_id]));

  set_callback(kernel_events[kernel_id], "ooo_queue");
  OCL_CHECK(err, err = kernel_events[kernel_id].wait());
}

void fpga_hlsinf(Tensor *input, Tensor *input_add, int H, int W, int Ichannels, int Ochannels, 
                 int KH, int KW, int SH, int SW, int PT, int PB, int PL, int PR, 
                 int enable_relu, float relu_factor, int enable_maxp, int enable_avgp,
                 int enable_clipping, int enable_shift, int pos_shift, int enable_add, int enable_stm, Tensor *filter, Tensor *bias, Tensor *output) {
  
  _debug_fpga_funcs("fpga_hlsinf");
  _profile_fpga(_FPGA_HLSINF, 0);
  _profile_fpga_tensor(input);
  if(enable_add)   _profile_fpga_tensor(input_add);
  _profile_fpga_tensor(filter);
  _profile_fpga_tensor(bias);
  int rows = H;
  int global_offset = 0;
  int min_clip = 0;
  int max_clip = 0;
  int dir_shift = 0;

  int num_kernels = k_conv2d_num_kernels;
  int CPI = k_conv2d_cpi;
  int CPO = k_conv2d_cpo;
  int HO = H;
  int WO = W;
#ifdef DEBUG_VERBOSE
  printf("HLSinf:  In=%3dx%3dx%3d, Out=%3dx%3dx%3d K=%1dx%1d S=%1dx%1d P=%1dx%1dx%1dx%1d RELU %d RELU_FACTOR %f MAXP %d AVGP %d CLIPPING %d SHIFT %d ADD %d STM %d\n", 
         Ichannels, H, W, Ochannels, HO, WO, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_add, enable_stm);
#endif
  // conv2D parameters
  cl::Buffer I     = *(cl::Buffer*)input->fpga_ptr;     // input activations
  cl::Buffer K     = *(cl::Buffer*)filter->fpga_ptr;    // kernel
  cl::Buffer B     = *(cl::Buffer*)bias->fpga_ptr;      // bias
  int use_bias     = 1;                                 // whether use bias or not
  cl::Buffer O     = *(cl::Buffer*)output->fpga_ptr;         // output activations
  cl::Buffer I_add;
  if (enable_add) {
    I_add = *(cl::Buffer*)input_add->fpga_ptr; // input add data
  } else {
    I_add = I;
  }

    PROFILING_HEADER(fpga_hlsinf);
  // Depending on the number of kernels available we split the convolution operation into multiple frames, and launch one thread per kernel
  if (num_kernels == 1) {
    // just one kernel which handles all the conv operation
    int first_o_iter = 0;
    int last_o_iter = ((Ochannels + (CPO-1)) / CPO) - 1;

    HLSinf_kernel(I, I_add, H, W, rows, PT, PB, PL, PR, SH, SW, Ichannels, Ochannels, first_o_iter,
                    last_o_iter, enable_relu, enable_stm, relu_factor, K, B, O, global_offset, enable_maxp,
                                enable_avgp, enable_clipping, enable_shift, enable_add, min_clip, max_clip,
                    dir_shift, pos_shift, CPI, CPO, 0);

  } else {
    // several kernels available, let's split the operation in sets of output channels
    int O_ITER = (Ochannels + (CPO-1)) / CPO;
    // let's compute number of channels per kernel and number of final kernels to launch
    int num_kernels_to_launch;
    int o_iter_per_kernel;
    if (O_ITER < num_kernels) {
      num_kernels_to_launch = 1;
      o_iter_per_kernel = O_ITER;
    } else {
      num_kernels_to_launch = num_kernels;
      o_iter_per_kernel = O_ITER / num_kernels;
    }

    // Let's launch the kernels
    #pragma omp parallel for
    for (int k=0; k<num_kernels_to_launch; k++) {
      int first_o_iter = o_iter_per_kernel * k;
      int last_o_iter = first_o_iter + o_iter_per_kernel - 1;
      HLSinf_kernel(I, I_add, H, W, rows, PT, PB, PL, PR, SH, SW, Ichannels, Ochannels, first_o_iter,
                      last_o_iter, enable_relu, enable_stm, relu_factor, K, B, O, global_offset, enable_maxp,
                                  enable_avgp, enable_clipping, enable_shift, enable_add, min_clip, max_clip,
                      dir_shift, pos_shift, CPI, CPO, 0);
    }
  }
  PROFILING_FOOTER(fpga_hlsinf);
  _profile_fpga_tensor(output);
  _profile_fpga_tensor_print(output);
  
}

#endif
