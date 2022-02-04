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

// S10MX included in common header file in stratix standalone development project
#include <CL/opencl.h>
#include <CL/cl_ext_intelfpga.h>
#include "AOCLUtils/aocl_utils.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
// -- end of S10MX 


#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/profiling.h"

extern vector<cl_event> kernel_events; // Kernel events (completion)

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
void HLSinf_launch_kernel(void *I, void *I_add, int H, int W, int HO, int WO, int rows, int PT, int PB, int PL, int PR, int SH, int SW, int Ichannels, int Ochannels,  
                          int first_o_iter, int last_o_iter, int enable_relu, int enable_stm, float relu_factor, int enable_batch_norm, void *K, void *B, void *BN_values, void *O, 
                          int read_offset, int write_offset, int enable_maxp, int enable_avgp, int enable_clipping,
                          int enable_shift, int enable_add, int enable_upscale, int min_clip, int max_clip, int dir_shift, int pos_shift, int CPI, int CPO, int kernel_id) {

  // Error variable
  cl_int err;

  // set kernel arguments
  int arg = 0;

  // input iterations
  int I_ITER = (Ichannels + (CPI-1)) / CPI;

    printf("This implementation only supports 1 kernel \n");
  printf("kernel_id: %d\n", kernel_id);
  #ifdef DEBUG_VERBOSE
  printf("I_ITER %d, first %d last %d enable_relu %d relu_factor %f enable_stm %d enable_maxp %d enable_avgp %d enable_clip %d min_clip %d max_clip %d enable_shift %d enable_add %d\n PT %d PB %d PL %d PR %d SH %d SW %d upscale %d batch_norm %d\n",
                  I_ITER, first_o_iter, last_o_iter, enable_relu, relu_factor, enable_stm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift,
                  enable_add, PT, PB, PL, PR, SH, SW, enable_upscale, enable_batch_norm);
  printf("H %d W %d rows %d Ich %d Och %d\n", H, W, rows, Ichannels, Ochannels);
  printf("min_clip %d max_clip %d, shifts %d, direction %d enable_upscale %d\n", min_clip, max_clip, pos_shift, dir_shift, enable_upscale);
  #endif

  OCL_CHECK(err, err = (kernel_id != 0));
  
  if (kernel_id == 0) {
    cl_event  kernel_events[K_SUBKERNELS];

    
    // Legacy num_kernels interpretation
    //uint num_kernels;
    //uint o_iter_per_kernel;

    //num_kernels = 1;
    //o_iter_per_kernel = o_iter;

    //uint k = 0; // (int k=0; k<num_kernels; k++)
    //uint o_iter_first = o_iter_per_kernel * k;
    //uint o_iter_last  = o_iter_first + o_iter_per_kernel - 1;
    // --------------------------------
    // renaming of pointers to cl mem objects in order to avoid eddl naming rules/dependencies


    // cl_uint k_H    = (cl_uint)H;
    // cl_uint k_W    = (cl_uint)W;
    // cl_uint k_rows = (cl_uint)rows;
    // cl_uint k_I    = (cl_uint)I_input;
    // cl_uint k_O    = (cl_uint)O_output;
    // cl_uint k_i_iter        = (cl_uint)i_iter;
    // cl_uint k_o_iter_first  = (cl_uint)o_iter_first;
    // cl_uint k_o_iter_last   = (cl_uint)o_iter_last;
    // cl_uint k_enable_relu   = (cl_uint)enable_relu;
    // cl_uint k_global_offset = (cl_uint)global_offset;
    // cl_uint k_enable_upper_padding = (cl_uint)enable_upper_padding;
    // cl_uint k_enable_lower_padding = (cl_uint)enable_lower_padding;
    // cl_uint k_enable_maxpooling    = (cl_uint)enable_maxpooling;
    // cl_uint k_enable_avgpooling    = (cl_uint)enable_avgpooling;
    // cl_uint k_enable_clipping = (cl_uint)enable_clipping;
    // cl_uint k_enable_shift    = (cl_uint)enable_shift;
    // cl_uint k_min_clip  = (cl_uint)min_clip;
    // cl_uint k_max_clip  = (cl_uint)max_clip;
    // cl_uint k_dir_shift = (cl_uint)dir_shift;
    // cl_uint k_pos_shift = (cl_uint)pos_shift;

    cl_mem  k_mem_I        = (cl_mem)I;
    cl_mem  k_mem_I_add    = (cl_mem)I_add;
    cl_uint k_H    = (cl_uint)H;
    cl_uint k_W    = (cl_uint)W;
    // HO
    // WO
    cl_uint k_rows    = (cl_uint)rows;
    // PT
    // PB
    // PL
    // PR
    // SH
    // SW
    cl_uint k_I    = (cl_uint)Ichannels;
    cl_uint k_O    = (cl_uint)Ochannels;
    
    cl_uint k_i_iter          = (cl_uint)I_ITER;
    cl_uint k_o_iter_first    = (cl_uint)first_o_iter;
    cl_uint k_o_iter_last     = (cl_uint)last_o_iter;
    cl_uint k_enable_relu     = (cl_uint)enable_relu;
    // enable_stm
    // relu_factor
    // enable_batch_norm
    cl_mem  k_mem_K    = (cl_mem)K;
    cl_mem  k_mem_B    = (cl_mem)B;
    cl_mem  k_mem_BN_values  = (cl_mem)BN_values;
    cl_mem  k_mem_O    = (cl_mem)O;
    // read_offset 
    // write_offset
    cl_uint k_global_offset = 0; //(cl_uint)global_offset;
    cl_uint k_enable_upper_padding = 1;//(cl_uint)enable_upper_padding;
    cl_uint k_enable_lower_padding = 1;//(cl_uint)enable_lower_padding;
    
    cl_uint k_enable_maxpooling    = (cl_uint)enable_maxp;
    cl_uint k_enable_avgpooling    = (cl_uint)enable_avgp;
    cl_uint k_enable_clipping = (cl_uint)enable_clipping;
    cl_uint k_enable_shift    = (cl_uint)enable_shift;
    // enable_add
    // enable_upscale
    cl_uint k_min_clip  = (cl_uint)min_clip;
    cl_uint k_max_clip  = (cl_uint)max_clip;
    cl_uint k_dir_shift = (cl_uint)dir_shift;
    cl_uint k_pos_shift = (cl_uint)pos_shift;


    cl_uint k_M = k_H * k_W * k_I;
    cl_uint k_N = k_H * k_W * k_O;


    cl_uint k_enable_pooling = (k_enable_maxpooling != 0) || (k_enable_avgpooling != 0);
    //----------------------------------------------------------

    //cl_uint k_offset_factor = ((I_input + CPI - 1) / CPI) * CPI * CPO * 9;
    //cl_uint k_offset_factor = ((I_input + CPI - 1) / CPI) * CPI * CPO;  // since we are reading kernel_t data type on the other side, we do not multiply *9 (kH*kW)
    cl_uint k_offset_factor = ((k_I + CPI - 1) / CPI) ;

    cl_uint k_O_ITER        = k_o_iter_last - k_o_iter_first + 1;

#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels for 2D convolution\n");
    printf("k_conv2D H %d   W %d   rows %d   I %d   O %d   I_ITER %d   o_iter_first %d   o_iter_last %d   enable_relu %d   "  \
        "   upper_padding %d   lower_padding %d, maxpooling %d   avgpooling %d   enable clip %d   enable shift %d \n",
        k_H, k_W, k_rows, k_I, k_O, k_i_iter, k_o_iter_first, k_o_iter_last, k_enable_relu, 
        k_enable_upper_padding, k_enable_lower_padding, k_enable_maxpooling, k_enable_avgpooling, k_enable_clipping, k_enable_shift 
        );
#endif

    //size_t window_size 

    // buffers already allocated and transferred

    cl_uint arg_ind = 0;
    //--------------------------------
    // let's set the kernels arguments
    // DATA IN
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_DATA_IN_READER");
#endif
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_mem),  (void*)&k_mem_I)); //(void*)&buffer_i));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_M));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_rows));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_global_offset));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_enable_lower_padding));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_enable_upper_padding));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_DATA_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));

    // KERNEL_IN
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_KERNEL_IN_READER");
#endif
    arg_ind = 0;
    //kernel void kernel_in(global kernel_t * kernel, uint offset_factor, uint I_ITER, uint o_iter_first, uint O_ITER){...}
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], arg_ind++, sizeof(cl_mem),  (void*)&k_mem_K)); //(void*)& buffer_k));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_offset_factor));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_o_iter_first));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    // BIAS_IN
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_BIAS_IN_READER");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_BIAS_IN_READER], arg_ind++, sizeof(cl_mem),  (void*)&k_mem_B)); //(void*)&buffer_bias));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_BIAS_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_o_iter_first));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_BIAS_IN_READER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    // WRITE
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_WRITER");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_mem),  (void*)&k_mem_O)); //(void*)&buffer_o));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_uint), (void*)&k_rows));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_uint), (void*)&k_o_iter_first));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_WRITER], arg_ind++, sizeof(cl_uint), (void*)&k_enable_pooling));

    // IB - (INPUT BUFFER)
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_INPUT_BUFFER");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_rows));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_enable_lower_padding));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_INPUT_BUFFER], arg_ind++, sizeof(cl_uint), (void*)&k_enable_upper_padding));

    // PAD - PADDING
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_PADDING");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_PADDING], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_PADDING], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_PADDING], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_PADDING], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_PADDING], arg_ind++, sizeof(cl_uint), (void*)&k_enable_lower_padding));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_PADDING], arg_ind++, sizeof(cl_uint), (void*)&k_enable_upper_padding));

    // CVT
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_CVT");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_rows));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    // MUL - MULTIPLIER
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_MULTIPLIER");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_MULTIPLIER], arg_ind++, sizeof(cl_uint), (void*)&k_rows));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_MULTIPLIER], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_MULTIPLIER], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_MULTIPLIER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    // ADD - ADDER
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_ADDER");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_ADDER], arg_ind++, sizeof(cl_uint), (void*)&k_rows));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_ADDER], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_ADDER], arg_ind++, sizeof(cl_uint), (void*)&k_i_iter));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_ADDER], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    // RELU
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_RELU");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_RELU], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_RELU], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_RELU], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_RELU], arg_ind++, sizeof(cl_uint), (void*)&k_enable_relu));

    // POOL CVT
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_POOL_CVT");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_enable_maxpooling));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_enable_avgpooling));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_CVT], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    // POOL POOLING
#ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: set kernel %s arguments\n", "K_POOL_POOLING");
#endif
    arg_ind = 0;
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_POOLING], arg_ind++, sizeof(cl_uint), (void*)&k_H));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_POOLING], arg_ind++, sizeof(cl_uint), (void*)&k_W));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_POOLING], arg_ind++, sizeof(cl_uint), (void*)&k_enable_maxpooling));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_POOLING], arg_ind++, sizeof(cl_uint), (void*)&k_enable_avgpooling));
    OCL_CHECK(err, err = clSetKernelArg(kernel_hlsinf[kernel_id][K_POOL_POOLING], arg_ind++, sizeof(cl_uint), (void*)&k_O_ITER));

    //

    cl_uint num_pixels_in  = H * W;
    cl_uint num_pixels_out = rows * W;
    // cl_uint num_kernels_in = 1; seguro que es uno ?
    // cl_uint num_bias_in    = ; a este que valor le toca ?


    //size_t sample_data_in_size   = num_pixels_in;
    //size_t sample_data_in_size   = num_pixels_in * k_O_ITER;
    size_t sample_data_out_size  = num_pixels_out;
    //  size_t sample_kernel_in_size = num_kernels_in;
    //  size_t sample_bias_in_size   = num_bias_in;

    // clEnqueueNDRangeKernel(q, kernel_conv2D[k], 1, NULL, &ws, &ls, 0, NULL, &kernel_events[k]);
    // cl_int clEnqueueNDRangeKernel(
    //        cl_command_queue command_queue,
    //        cl_kernel kernel,
    //        cl_uint work_dim,
    //        const size_t* global_work_offset,
    //        const size_t* global_work_size,
    //        const size_t* local_work_size,
    //        cl_uint num_events_in_wait_list,
    //        const cl_event* event_wait_list,
    //        cl_event* event
    //       );
    // clEnqueueNDRangeKernel( command_queue,           kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    //clEnqueueNDRangeKernel(              q, kernel_conv2D[k],         1,              NULL,              &ws,             &ls,                       0, NULL, &kernel_events[k]);
    //   queues[K_DATA_IN_READER], kernels[K_DATA_IN_READER], 1, NULL, &sample_data_in_size, NULL, 0, NULL, NULL             );         
    //--------------------------------
    size_t ws = 1;
    size_t ls = 1;
    // Let's trigger kernels execution
    //double time = getCurrentTimestamp();

    #ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: Start kernels\n");
    #endif
    // read data_in
    //OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_DATA_IN_READER], 1, NULL, &sample_data_in_size, NULL, 0, NULL, NULL));
    #ifdef DEBUG_VERBOSE 
    printf("                 %s\n", subkernel_names[K_DATA_IN_READER]);
    #endif
    OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_DATA_IN_READER], 1, NULL, &ws, &ls, 0, NULL, &kernel_events[K_DATA_IN_READER]));//NULL));
    // read kernel_in
    //OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], 1, NULL, &sample_kernel_in_size, NULL, 0, NULL, NULL));
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_KERNEL_IN_READER]);
    #endif
    OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], 1, NULL, &ws, &ls, 0, NULL, &kernel_events[K_KERNEL_IN_READER]));// NULL));
    // read bias_in
    //OCL_CHECK(err, err = clEnqueueNDRangeKernel(q], kernel_hlsinf[kernel_id][K_KERNEL_IN_READER], 1, NULL, &sample_bias_in_size, NULL, 0, NULL, NULL));
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_BIAS_IN_READER]);
    #endif
    OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_BIAS_IN_READER], 1, NULL, &ws, &ls, 0, NULL, &kernel_events[K_BIAS_IN_READER]));// NULL));

    // ib - input buffer
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_INPUT_BUFFER]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_INPUT_BUFFER], 0, NULL, &kernel_events[K_INPUT_BUFFER]));// NULL));

    // padding
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_PADDING]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_PADDING], 0, NULL, &kernel_events[K_PADDING]));// NULL));

    // cvt
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_CVT]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_CVT], 0, NULL, &kernel_events[K_CVT]));// NULL));

    // mul
    #ifdef DEBUG_HOSkernel_eventsT_KERNELS
    printf("                 %s\n", subkernel_names[K_MULTIPLIER]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_MULTIPLIER], 0, NULL, &kernel_events[K_MULTIPLIER]));// NULL));

    //add
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_ADDER]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_ADDER], 0, NULL, &kernel_events[K_ADDER]));// NULL));

    // relu
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_RELU]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_RELU], 0, NULL, &kernel_events[K_RELU]));// NULL));

    // pool_cvt
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_POOL_CVT]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_POOL_CVT], 0, NULL, &kernel_events[K_POOL_CVT]));// NULL));

    // pool_pooling 
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_POOL_POOLING]);
    #endif
    OCL_CHECK(err, err = clEnqueueTask(q, kernel_hlsinf[kernel_id][K_POOL_POOLING], 0, NULL, &kernel_events[K_POOL_POOLING]));// NULL));

    // write
    //OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_WRITER], 1, NULL,  &sample_data_out_size, NULL, 0, NULL, NULL));
    #ifdef DEBUG_VERBOSE
    printf("                 %s\n", subkernel_names[K_WRITER]);
    #endif
    OCL_CHECK(err, err = clEnqueueNDRangeKernel(q, kernel_hlsinf[kernel_id][K_WRITER], 1, NULL,  &ws, &ls, 0, NULL, &kernel_events[K_WRITER]));// NULL));

    #ifdef DEBUG_VERBOSE
    printf("run_aoc_kernels: all kernels launched\n");
    #endif
    printf("run_aoc_kernels: waiting for kernels completion\n");

    // Wait for command queue to complete pending events
    //for(int i=0; i<K_SUBKERNELS; ++i) {
    OCL_CHECK(err, err = clFinish(q));
    //}

    // getCurrentTimestamp is a helper function (common) that returns time in seconds
    // Record execution time
    //time = getCurrentTimestamp() - time;

    //kernels_execution_time = time * 1E6; // value in us

    //int num_pixels = H*W;
    //double gpixels_per_sec = ((double)(num_pixels / time) * 1E-9);
    //printf("\tThroughput = %.4f Gpx / sec\n", gpixels_per_sec);

    //printf("\tProcessing time = %.4f ms\n", (float)(time * 1E3));

    // OpenCL kernel time
    //printf("JM10 kernel profiler timings summary\n");
    #ifdef DEBUG_VERBOSE
    cl_ulong ts_first;
    cl_ulong te_last;
    cl_ulong ts_writer;
    cl_ulong te_writer;
    cl_ulong diff_writer;
    cl_ulong diff_kernels;
    OCL_CHECK(err, err = clGetEventProfilingInfo(kernel_events[0], CL_PROFILING_COMMAND_START, sizeof(ts_first), &ts_first, NULL));
    OCL_CHECK(err, err = clGetEventProfilingInfo(kernel_events[K_SUBKERNELS-1], CL_PROFILING_COMMAND_END, sizeof(te_last), &te_last, NULL));
    OCL_CHECK(err, err = clGetEventProfilingInfo(kernel_events[K_WRITER], CL_PROFILING_COMMAND_START, sizeof(ts_writer), &ts_writer, NULL));
    OCL_CHECK(err, err = clGetEventProfilingInfo(kernel_events[K_WRITER], CL_PROFILING_COMMAND_END, sizeof(te_writer), &te_writer, NULL));

    // profiling info is returned in ns
    diff_writer  = te_writer - ts_writer;
    diff_kernels = te_last   - ts_first;

    printf("PROFILE EVENT - TIME      WRITER KERNEL = %lu ns  (%lf ms)\n",  diff_writer,((double)diff_writer/(double)1000000.0));
    printf("PROFILE EVENT - TIME LAST -FIRST KERNEL = %lu ns  (%lf ms)\n",  diff_kernels,((double)diff_kernels/(double)1000000.0));

    // update kernels execution time
    double kernels_execution_time;
    kernels_execution_time = diff_kernels / 1000.0; // value from ns to us
    //printf("test_kernels: update kernels execution time with event profiling time, %lf ns to %lu ns \n",  kernels_execution_time, diff_kernels);
    printf("\tProcessing time = %.4f ms\n", (float)(kernels_execution_time/1000.0));
    #endif


    //--------------------------------
    //

  }

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
                   int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_add, int enable_stm, int enable_upscale, Tensor *filter, Tensor *bias, Tensor *batch_norm_values, Tensor *output, int read_offset, int write_offset, int rows, int HO, int WO) {

  // accelerator geometry
  int num_kernels = hlsinf_num_kernels;
  int CPI = hlsinf_cpi;
  int CPO = hlsinf_cpo;

  #ifdef DEBUG_VERBOSE
  printf("HLSinf:  In=%3dx%3dx%3d, Out=%3dx%3dx%3d K=%1dx%1d S=%1dx%1d P=%1dx%1dx%1dx%1d RELU %d RELU_FACTOR %f MAXP %d AVGP %d CLIPPING %d MINCLIP %d MAXCLIP %d SHIFT %d ADD %d STM %d UPSCALE %d\n",
         Ichannels, H, W, Ochannels, HO, WO, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift, enable_add, enable_stm, enable_upscale);
  #endif

  // arguments
  void * I     = input->fpga_ptr;     // input activations
  void * K     = filter->fpga_ptr;    // kernel
  void * B     = bias->fpga_ptr;      // bias
  int use_bias     = 1;                                 // whether use bias or not
  void * O     = output->fpga_ptr;         // output activations
  void * I_add; void *BN_values; 
  if (enable_add) {
    I_add = input_add->fpga_ptr; // input add data
  } else {
    I_add = I;
  }

  if (enable_batch_norm) {
    BN_values = batch_norm_values->fpga_ptr; // ERROR: no tiene fpga_ptr porque es puntero CPU
  } else {
    BN_values = I;
  }

  PROFILING_HEADER(fpga_hlsinf);
  // Depending on the number of kernels available we split the convolution operation into multiple frames, and launch one thread per kernel
  if (num_kernels == 1) {
    // just one kernel which handles all the conv operation
    int first_o_iter = 0;
    int last_o_iter = ((Ochannels + (CPO-1)) / CPO) - 1;

    HLSinf_launch_kernel(I, I_add, H, W, HO, WO, rows, PT, PB, PL, PR, SH, SW, Ichannels, Ochannels, first_o_iter, last_o_iter, enable_relu, enable_stm, relu_factor, enable_batch_norm, K, B, BN_values, O, read_offset, write_offset, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_add, enable_upscale, min_clip, max_clip, dir_shift, pos_shift, CPI, CPO, 0);

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
#ifdef DEBUG_VERBOSE
      printf("Kernel %d first iter %d last %d\n", k, first_o_iter, last_o_iter);
#endif
      HLSinf_launch_kernel(I, I_add, H, W, HO, WO, rows, PT, PB, PL, PR, SH, SW, Ichannels, Ochannels, first_o_iter, last_o_iter, enable_relu, enable_stm, relu_factor, enable_batch_norm, K, B, BN_values, O, read_offset, write_offset, enable_maxp, enable_avgp, enable_clipping, enable_shift, enable_add, enable_upscale, min_clip, max_clip, dir_shift, pos_shift, CPI, CPO, k);
    }
  }
  PROFILING_FOOTER(fpga_hlsinf);

}

// ---------------------------------------------------------------------
// fpga_hlsinf
void fpga_hlsinf(Tensor *input, Tensor *input_add, int H, int W, int Ichannels, int Ochannels, int KH, int KW, int SH, int SW, int PT, int PB, int PL, int PR, 
                 int enable_relu, float relu_factor, int enable_batch_norm, int enable_maxp, int enable_avgp, int enable_clipping, int min_clip, int max_clip, 
                 int enable_shift, int pos_shift, int dir_shift, int enable_add, int enable_stm, int enable_upscale, Tensor *filter, Tensor *bias, Tensor *batch_norm_values, Tensor *output) {
 
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
  if(enable_add)   _profile_fpga_tensor("  input add: ", input_add, hlsinf_input_format);
  _profile_fpga_tensor("  filter  ", filter, hlsinf_filter_format);
  _profile_fpga_tensor("  bias    ", bias, hlsinf_bias_format);
  if(enable_batch_norm)  {
    _profile_fpga_tensor("  bn_v    ", batch_norm_values, hlsinf_output_format);
  }

  // output geometry
  int HO;
  int WO;
  int HO_conv = (H + PT + PB - (KH - 1)) / SH;
  int WO_conv = (W + PL + PR - (KW - 1)) / SW;
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

      //printf("H %d W %d Padding %d %d %d %d read offset %d write offset %d rows to read %d HO_conv %d WO_conv %d HO %d WO %d\n", H, W, PT_frame, PB_frame, PL_frame, PR_frame, read_offset_frame, write_offset_frame, rows_to_read, HO_conv, WO_conv, HO, WO);

      // run kernel
      HLSinf_launch(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT_frame, PB_frame, PL_frame, PR_frame, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip,enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_upscale, filter, bias, batch_norm_values, output, 
		        read_offset_frame, write_offset_frame, rows_to_read, HO, WO);
    }
  } else {
    // single frame operation
    HLSinf_launch(input, input_add, H, W, Ichannels, Ochannels, KH, KW, SH, SW, PT, PB, PL, PR, enable_relu, relu_factor, enable_batch_norm, enable_maxp, enable_avgp, enable_clipping, min_clip, max_clip, enable_shift, pos_shift, dir_shift, enable_add, enable_stm, enable_upscale, filter, bias, batch_norm_values, output, 0, 0, H, HO, WO);
  }

  gettimeofday(&time2, NULL);
  unsigned long long t = ((time2.tv_sec - time1.tv_sec) * 1000000) + (time2.tv_usec - time1.tv_usec);
  #ifdef DEBUG_FPGA 
  printf("HLSinf: Time %llu us - %0dx%0dx%0dx%0d\n", t, Ochannels, Ichannels, H, W);
  #endif

  // profiling
  _profile_fpga_tensor("  output  ", output, hlsinf_output_format);
  _profile_fpga_tensor_print(output);
  /*int size = output->size;
  unsigned char *b = (unsigned char *)malloc(size);
  fpga_copy_memory_from_fpga((cl::Buffer*)output->fpga_ptr, b, size);
  for (int x=0; x<size; x++) {
      printf("%d ", b[x]);
      printf("\n");
  }
  free(b);*/
  /*fpga_write_buffer("input.bin", input->fpga_ptr, input->size, 1);
  fpga_write_buffer("weights.bin", filter->fpga_ptr, filter->size, 1);
  fpga_write_buffer("bias.bin", bias->fpga_ptr, bias->size, 4);
  fpga_write_buffer("output.bin", output->fpga_ptr, output->size, 1);
  exit(1);*/
  _profile_fpga(_FPGA_HLSINF, 1);
}

#endif
