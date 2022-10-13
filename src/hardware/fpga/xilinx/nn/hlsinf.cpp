/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include <sys/time.h>


#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/profiling.h"

extern vector<cl::Event> kernel_events; // Kernel events (completion)
#if defined(WRITE_TENSORS_TO_FILE) || defined(WRITE_TENSORS_TO_FILE_ASCI)
int id_write_tensors_to_file = 0;
#endif

PROFILING_ENABLE_EXTERN(fpga_hlsinf);

// -----------------------------------------------------------------
// HLSinf 
//
//
//
//

// -----------------------------------------------------------------------------------
// HLSinf_launch_kernel
//
// This function launches one HLSinf kernel
//
// Arguments:
//   I: Input buffer (OpenCL buffer)
//   I_add: Input buffer (OpenCL buffer) for Add layer
//   H, W: Height and Width of input data channels
//   rows: Number of rows to read from the input
//   PT, PB, PL, PR: Top, Bottom, Left, and Right padding for Convolution layer
//   SH, SW: Vertical and Horizontal stride for Convolution layer
//   Ichannels: Number of input channels
//   Ochannels: Number of output channels
//   first_o_iter: First output iteration to compute
//   last_o_iter: Last output iteration to compute
//   enable_relu: Activates the ReLU layer
//   enable_stm: Activates the STM layer (Sigmoid + Tanh + Multiply)
//   relu_factor: Factor to apply to the ReLU layer (Leaky ReLU)
//   K, B, O: Filter, Bias, Output buffers (OpenCL buffers)
//   read_offset: Offset for the input data
//   write_offset: Offset for the output data
//   enable_maxp: Activates the Maxpooling layer
//   enable_avgp: Activates the Avgpooling layer
//   enable_clipping: Activates the Clipping layer
//   enable_shift: Activates the shift layer
//   enable_add: Activates the Add layer
//   min_clip: Minimum clipping value
//   max_clip: Maximum clipping value
//   dir_shift: Direction for shift Layer
//   pos_shift: Number of bit shift for shift layer
//   CPI: Kernel channels per input
//   CPO: Kernel channels per output
//   kernel_id: Kernel ID (which kernel to use in a multi-kernel setting)
//
void HLSinf_launch_kernel(cl::Buffer I, cl::Buffer I_add, int H, int W, int HO, int WO, int rows, int PT, int PB, int PL, int PR, int SH, int SW, int Ichannels, int Ochannels,  
                          int first_o_iter, int last_o_iter, int enable_relu, int enable_stm, float relu_factor, int enable_batch_norm, cl::Buffer K, cl::Buffer B, cl::Buffer BN_values, cl::Buffer O, 
                          int read_offset, int write_offset, int enable_maxp, int enable_avgp, int enable_clipping,
                          int enable_shift, int enable_add, int enable_upscale, int use_weight_buffer, int first_row_weight_buffer, int weight_buffer_initialized, int min_clip, int max_clip, int dir_shift, int pos_shift, int CPI, int CPO, int kernel_id) {

  // Error variable
  cl_int err;

  int write_to_weight_buffer = use_weight_buffer && !weight_buffer_initialized;
  int read_from_weight_buffer = use_weight_buffer && weight_buffer_initialized;
  int read_from_mem = 1;
  int read_from_b0 = 0;
  int read_from_b1 = 0;
  int write_to_mem = 1;
  int write_to_b0 = 0;
  int write_to_b1 = 0;

  // set kernel arguments
  int arg = 0;

  // input iterations
  int I_ITER = (Ichannels + (CPI-1)) / CPI;

#ifdef DEBUG_VERBOSE
  printf("I_ITER %d, first %d last %d enable_relu %d relu_factor %f enable_stm %d enable_maxp %d enable_avgp %d enable_clip %d min_clip %d max_clip %d enable_shift %d enable_add %d\n PT %d PB %d PL %d PR %d SH %d SW %d upscale %d batch_norm %d\n",
                  I_ITER, first_o_iter, last_o_iter, enable_relu, relu_factor, enable_stm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift,
                  enable_add, PT, PB, PL, PR, SH, SW, enable_upscale, enable_batch_norm);
  printf("H %d W %d rows %d Ich %d Och %d\n", H, W, rows, Ichannels, Ochannels);
  printf("min_clip %d max_clip %d, shifts %d, direction %d enable_upscale %d\n", min_clip, max_clip, pos_shift, dir_shift, enable_upscale);
#endif

  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, I));
  if (hlsinf_add_support) OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, I_add));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, H));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, W));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, HO));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, WO));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, rows));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, PT));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, PB));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, PL));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, PR));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, SH));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, SW));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, Ichannels));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, Ochannels));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, first_o_iter));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, last_o_iter));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_relu));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_stm));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, relu_factor));
  if (hlsinf_bn_support) OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_batch_norm));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, K));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, B));
  if (hlsinf_bn_support) OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, BN_values));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, O));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, read_offset));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, write_offset));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_maxp));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_avgp));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_clipping));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_shift));
  if (hlsinf_add_support) OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_add));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, min_clip));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, max_clip));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, dir_shift));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, pos_shift));
  OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, enable_upscale));  // upsize
  if (hlsinf_weight_buffer != 0) {
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, write_to_weight_buffer));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, read_from_weight_buffer));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, first_row_weight_buffer));
  }
  if (hlsinf_data_buffer != 0) {
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, read_from_mem));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, read_from_b0));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, read_from_b1));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, write_to_mem));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, write_to_b0));
    OCL_CHECK(err, err = kernel_hlsinf[kernel_id].setArg(arg++, write_to_b1));
  }

  // Launch the Kernel
  OCL_CHECK(err, err = (*q).enqueueNDRangeKernel(kernel_hlsinf[kernel_id], 0, 1, 1, NULL, &kernel_events[kernel_id]));

  set_callback(kernel_events[kernel_id], "ooo_queue");
  OCL_CHECK(err, err = kernel_events[kernel_id].wait());
}

// ---------------------------------------------------------------------
// HLSinf_launch
//
// This function launches the HLSinf kernel for the given parameters. It 
// performs the requested operation by using all the HLSinf accelerators
// implemented on the FPGA.
//
// Parameters:
//   - input: Input tensor object (input data)
//   - input_add: Input tensor object (input data) for add layer
//   - H, W: Height and Width of each input channel
//   - Ichannels: Number of input channels
//   - Ochannels: Number of output channels
//   - KH, KW: Height and Width of convolution filters
//   - SH, SW: Vertical and Horizontal stride for convolution operation
//   - PT, PB, PL, PR: Top, Bottom, Left, and Right padding for convolution operation
//   - enable_relu: Activates the ReLU layer
//   - relu_factor: Factor to apply to the ReLU layer (Leaky ReLU)
//   - enable_maxp: Activates the Maxpooling layer
//   - enable_avgp: Activates the Avgpooling layer
//   - enable_clipping: Activates the Clipping layer
//   - enable_shift: Activates the Shift layer
//   - pos_shift: Number of bit shifts
//   - enable_add: Activates the Add layer
//   - enable_stm: Activates the STM layer (Sigmoid + Tanh + Multiply)
//   - filter: Filter tensor object (filters)
//   - bias: Bias tensor object (bias)
//   - output: Output tensor object (output data)
//   - read_offset: Offset for reading input data
//   - write_offset: Offset for writting output data
//   - rows: Number of rows to read for input data
//   - HO, WO: Height and Width of output data channel
//
//
// If more than one accelerator is present on the FPGA, this function splits
// the operation in disjoint output channels, thus each accelerator computes
// in parallel a disjoint set of output channels
//
void HLSinf_launch(Tensor *input, Tensor *input_add, int H, int W, int Ichannels, int Ochannels, int KH, int KW, int SH, int SW, int PT, int PB, int PL, int PR, int enable_relu, float relu_factor, int enable_batch_norm, int enable_maxp, int enable_avgp,
                   int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_add, int enable_stm, int enable_upscale, int use_weight_buffer, int first_row_weight_buffer, int weight_buffer_initialized, Tensor *filter, Tensor *bias, Tensor *batch_norm_values, Tensor *output, int read_offset, int write_offset, int rows, int HO, int WO) {

  // accelerator geometry
  int num_kernels = hlsinf_num_kernels;
  int CPI = hlsinf_cpi;
  int CPO = hlsinf_cpo;

  //#ifdef FPGA_DEBUG
  //printf("HLSinf:  In=%3dx%3dx%3d, Out=%3dx%3dx%3d K=%1dx%1d S=%1dx%1d P=%1dx%1dx%1dx%1d RELU %d RELU_FACTOR %f MAXP %d AVGP %d CLIPPING %d MINCLIP %d MAXCLIP %d SHIFT %d ADD %d STM %d UPSCALE %d\n",
  //       Ichannels, H, W, Ochannels, HO, WO, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift, enable_add, enable_stm, enable_upscale);
  //#endif

  // arguments
  cl::Buffer I     = *(cl::Buffer*)input->fpga_ptr;     // input activations
  cl::Buffer K     = *(cl::Buffer*)filter->fpga_ptr;    // kernel
  cl::Buffer B     = *(cl::Buffer*)bias->fpga_ptr;      // bias
  int use_bias     = 1;                                 // whether use bias or not
  cl::Buffer O     = *(cl::Buffer*)output->fpga_ptr;         // output activations
  cl::Buffer I_add, BN_values; 
  if (enable_add) {
    I_add = *(cl::Buffer*)input_add->fpga_ptr; // input add data
  } else {
    I_add = I;
  }

  if (enable_batch_norm) {
    BN_values = *(cl::Buffer*)batch_norm_values->fpga_ptr; // ERROR: no tiene fpga_ptr porque es puntero CPU
  } else {
    BN_values = I;
  }

  PROFILING_HEADER(fpga_hlsinf);
  // Depending on the number of kernels available we split the convolution operation into multiple frames, and launch one thread per kernel
  if (num_kernels == 1) {
    // just one kernel which handles all the conv operation
    int first_o_iter = 0;
    int last_o_iter = ((Ochannels + (CPO-1)) / CPO) - 1;

    HLSinf_launch_kernel(I, I_add, H, W, HO, WO, rows, PT, PB, PL, PR, SH, SW, Ichannels, Ochannels, first_o_iter, last_o_iter, enable_relu, enable_stm, relu_factor, enable_batch_norm, K, B, BN_values, O, read_offset, write_offset, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_add, enable_upscale, use_weight_buffer, first_row_weight_buffer, weight_buffer_initialized, min_clip, max_clip, dir_shift, pos_shift, CPI, CPO, 0);

  } else {
    // several kernels available, let's split the operation in sets of output channels
    int O_ITER = (Ochannels + (CPO-1)) / CPO;
    // let's compute number of channels per kernel and number of final kernels to launch
    int num_kernels_to_launch;
    int o_iter_per_kernel;
    int extra_iter = 0;

    if (O_ITER < num_kernels) {
      num_kernels_to_launch = 1;
      o_iter_per_kernel = O_ITER;
    } else {
      num_kernels_to_launch = num_kernels;
      o_iter_per_kernel = O_ITER / num_kernels;
      if(O_ITER > o_iter_per_kernel * num_kernels) extra_iter = O_ITER - o_iter_per_kernel * num_kernels;
    }

    // Let's launch the kernels
    #pragma omp parallel for
    for (int k=0; k<num_kernels_to_launch; k++) {
      int first_o_iter, last_o_iter;
      if(k == 0) {
        first_o_iter = o_iter_per_kernel * k;
        last_o_iter = first_o_iter + o_iter_per_kernel + extra_iter - 1;
      } else {
        first_o_iter = o_iter_per_kernel * k + extra_iter ;
        last_o_iter = first_o_iter + o_iter_per_kernel - 1;
      }
      HLSinf_launch_kernel(I, I_add, H, W, HO, WO, rows, PT, PB, PL, PR, SH, SW, Ichannels, Ochannels, first_o_iter, last_o_iter, enable_relu, enable_stm, relu_factor, enable_batch_norm, K, B, BN_values, O, read_offset, write_offset, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_add, enable_upscale, use_weight_buffer, first_row_weight_buffer, weight_buffer_initialized, min_clip, max_clip, dir_shift, pos_shift, CPI, CPO, k);
    }
  }
  PROFILING_FOOTER(fpga_hlsinf);

}

void fpga_hlsinf(Tensor *input, Tensor *input_add, int H, int W, int Ichannels, int Ochannels, int KH, int KW, int SH, int SW, int PT, int PB, int PL, int PR, 
                 int enable_relu, float relu_factor, int enable_batch_norm, int enable_maxp, int enable_avgp, int enable_clipping, int min_clip, int max_clip, 
                 int enable_shift, int pos_shift, int dir_shift, int enable_add, int enable_stm, int enable_upscale, int use_weight_buffer, int first_row_weight_buffer, int weight_buffer_initialized, Tensor *filter, Tensor *bias, Tensor *batch_norm_values, Tensor *output) {
 
  // profiling and debug	
  _profile_fpga(_FPGA_HLSINF, 0);

  // get load statistics
  struct timeval time1, time2;

  gettimeofday(&time1, NULL);

  #ifdef FPGA_DEBUG
  printf("HLSinf\n");
  printf("  params: %0dx%0dx%0dx%0d (KHxKW: %0dx%0d, PAD: %0d-%0d-%0d-%0d, SHxSW: %0dx%0d). ReLU %d, ReLU factor %f, Maxp %d, AvgP %d, Clip %d, min_clip %d, max_clip %d, Shift %d, bit shifts %d, dir_shift %d, Add %d, STM %d BN %d UPSCALE %d\n", 
                   Ochannels, Ichannels, H, W, KH, KW, PT, PB, PL, PR, SH, SW, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_batch_norm, enable_upscale);
  #endif
                        _profile_fpga_tensor("  input   ", input, hlsinf_input_format);
  if(enable_add)        _profile_fpga_tensor("  input add: ", input_add, hlsinf_input_format);
                        _profile_fpga_tensor("  filter  ", filter, hlsinf_filter_format);
                        _profile_fpga_tensor("  bias    ", bias, hlsinf_bias_format);
  if(enable_batch_norm) _profile_fpga_tensor("  bn_v    ", batch_norm_values, hlsinf_output_format);

  // output geometry
  int HO;
  int WO;
  int HO_conv = (H + PT + PB - KH + SH) / SH;
  int WO_conv = (W + PL + PR - KW + SW) / SW;
  if (enable_maxp || enable_avgp) {
    HO = HO_conv / 2;
    WO = WO_conv / 2;
  } else {
    HO = HO_conv;
    WO = WO_conv;
  }
  
  // HLSinf kernel limitations
  int HO_MAX = hlsinf_ho_max;
  int WO_MAX = hlsinf_wo_max;
  if (WO_conv > WO_MAX) {printf("Error, HLSinf kernel does not support output width larger than %d (WO = %d)\n", WO_MAX, WO); exit(1);}

  if (HO_conv > HO_MAX) {
    // We perform the convolution by spliting the problem into frames
    int num_frames = ceil( (float) HO_conv / (float) HO_MAX);

    for (int fr = 0; fr < num_frames; fr++) {

      // first output row for this frame
      int row_o = fr * HO_MAX;

      // rows to be produced in this frame
      int output_rows_frame = HO_MAX;
      if ((fr == num_frames-1) && ((HO_MAX * num_frames) != HO)) output_rows_frame = HO_conv % HO_MAX;

      // padding
      int PT_frame = (fr==0) ? PT : 0;
      int PB_frame = (fr == num_frames - 1) ? PB : 0;
      int PL_frame = PL;
      int PR_frame = PR;

      // first input row to read for this frame
      // row_i = (row_o * SH) - PT
      // We correct if negative (because of padding)
      //
      int row_i = (row_o * SH) - PT;
      if (row_i < 0) row_i = 0;
 
      // rows to read for this frame
      int rows_to_read = KH + ((output_rows_frame-1) * SH) - PT_frame - PB_frame;

      // read and write offsets
      int read_offset_frame = row_i * W;
      int write_offset_frame = (fr * HO_MAX * WO);

      #ifdef FPGA_DEBUG
      printf("H %d W %d Padding %d %d %d %d read offset %d write offset %d rows to read %d HO_conv %d WO_conv %d HO %d WO %d\n", H, W, PT_frame, PB_frame, PL_frame, PR_frame, read_offset_frame, write_offset_frame, rows_to_read, HO_conv, WO_conv, HO, WO);
      #endif

      // run kernel
      HLSinf_launch(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT_frame, PB_frame, PL_frame, PR_frame, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip,enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_upscale, 
            use_weight_buffer, first_row_weight_buffer, weight_buffer_initialized,
            filter, bias, batch_norm_values, output, 
		        read_offset_frame, write_offset_frame, rows_to_read, HO, WO);
    }
  } else {
    // single frame operation
    HLSinf_launch(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_upscale, use_weight_buffer, first_row_weight_buffer, weight_buffer_initialized, filter, bias, batch_norm_values, output, 0, 0, H, HO, WO);
  }

  gettimeofday(&time2, NULL);
  unsigned long long t = ((time2.tv_sec - time1.tv_sec) * 1000000) + (time2.tv_usec - time1.tv_usec);
  #ifdef DEBUG_FPGA 
  printf("HLSinf: Time %llu us - %0dx%0dx%0dx%0d\n", t, Ochannels, Ichannels, H, W);
  #endif

  // profiling
  _profile_fpga_tensor("  output  ", output, hlsinf_output_format);
  _profile_fpga_tensor_print(output);
  _profile_fpga(_FPGA_HLSINF, 1);

  #ifdef WRITE_TENSORS_TO_FILE
  char dummy[100];
  sprintf(dummy, "input_%03d.bin", id_write_tensors_to_file);
  fpga_write_buffer(dummy, input->fpga_ptr, input->size, hlsinf_input_format);
  sprintf(dummy, "weights_%03d.bin", id_write_tensors_to_file);
  fpga_write_buffer(dummy, filter->fpga_ptr, filter->size, hlsinf_filter_format);
  sprintf(dummy, "bias_%03d.bin", id_write_tensors_to_file);
  fpga_write_buffer(dummy, bias->fpga_ptr, bias->size, hlsinf_bias_format);
  if (enable_add) {
    sprintf(dummy, "add_%03d.bin", id_write_tensors_to_file);
    fpga_write_buffer(dummy, input_add->fpga_ptr, input_add->size, hlsinf_input_format);
  }
  if (enable_batch_norm) {
    sprintf(dummy, "batchnorm_%03d.bin", id_write_tensors_to_file);
    fpga_write_buffer(dummy, batch_norm_values->fpga_ptr, batch_norm_values->size, hlsinf_input_format);
  }
  sprintf(dummy, "output_%03d.bin", id_write_tensors_to_file);
  fpga_write_buffer(dummy, output->fpga_ptr, output->size, hlsinf_output_format);
  id_write_tensors_to_file++;
  #endif

  #ifdef WRITE_TENSORS_TO_FILE_ASCI
  char dummy[100];
  sprintf(dummy, "input_%03d.txt", id_write_tensors_to_file);
  fpga_write_buffer(dummy, input, input->size, hlsinf_input_format);
  //sprintf(dummy, "weights_%03d.txt", id_write_tensors_to_file);
  //fpga_write_buffer(dummy, filter, filter->size, hlsinf_filter_format); //commented because fpga_write_buffer performs the transpose operation -  GHW CHW
  //sprintf(dummy, "bias_%03d.txt", id_write_tensors_to_file);     
  //fpga_write_buffer(dummy, bias, bias->size, hlsinf_bias_format); //commented because fpga_write_buffer performs the transpose operation -  GHW CHW
  if (enable_add) {
    sprintf(dummy, "add_%03d.txt", id_write_tensors_to_file);
    fpga_write_buffer(dummy, input_add, input_add->size, hlsinf_input_format);
  }
  if (enable_batch_norm) {
    sprintf(dummy, "batchnorm_%03d.txt", id_write_tensors_to_file);
    fpga_write_buffer(dummy, batch_norm_values, batch_norm_values->size, hlsinf_input_format);
  }
  //sprintf(dummy, "output_%03d.txt", id_write_tensors_to_file);
  //fpga_write_buffer(dummy, output, output->size, hlsinf_output_format);
  id_write_tensors_to_file++;
  #endif

}

#endif
