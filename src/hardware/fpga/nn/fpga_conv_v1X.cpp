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

extern vector<cl::Event> kernel_events; // Kernel events (completion)


// -----------------------------------------------------------------
// conv2D version 1X of kernels
//
//

PROFILING_ENABLE_EXTERN(fpga_Conv2D);

// fpga_conv2D_v1X_kernel: This function launches the kernel execution on the FPGA
void fpga_conv2D_v1X_kernel(cl::Buffer I, int Irows, int Icols, int Ichannels, int num_rows, 
		        int enable_upper_padding, int enable_lower_padding, int global_offset, 
			cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels, int apply_relu, int CPI, int CPO, int kernel_id) {

  PROFILING_HEADER(fpga_Conv2D);

  int KW = 3;                   // kernel width
  int KH = 3;                   // kernel height
  int H = Irows;                // input channel height
  int W = Icols;                // input channel width

  // Error variable
  cl_int err;
  
  // iterations
  int I_ITER = (Ichannels + (CPI-1)) / CPI;
  int O_ITER = (Ochannels + (CPO-1)) / CPO;

  // set kernel arguments
  int arg = 0;
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Irows));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Icols));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, num_rows)); 
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Ichannels));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Ochannels));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, O_ITER));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, apply_relu)); // relu
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, K));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, B));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, O));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, global_offset));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_upper_padding));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_lower_padding));

  // Launch the Kernel
  OCL_CHECK(err, err = (*q).enqueueNDRangeKernel(kernel_conv2D[kernel_id], 0, 1, 1, NULL, &kernel_events[kernel_id]));

  set_callback(kernel_events[kernel_id], "ooo_queue");
  OCL_CHECK(err, err = kernel_events[kernel_id].wait());
  
  PROFILING_FOOTER(fpga_Conv2D);
}

// fgpa_conv2D_v1X_launch: This function launches all the possible conv2D kernels
void fpga_conv2D_v1X_launch(cl::Buffer I, int Irows, int Icols, int Ichannels, cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels, int apply_relu, int CPI, int CPO, int num_kernels, int max_rows) {

  // Depending on the number of kernels available we split the convolution operation into multiple frames, and launch one thread per kernel
  if (num_kernels == 1) {
    // just one kernel which handles all the conv operation
    int num_rows = Irows;
    int enable_upper_padding = 1;
    int enable_lower_padding = 1;
    int global_offset = 0;
    int kernel_id = 0;
    fpga_conv2D_v1X_kernel(I, Irows, Icols, Ichannels, num_rows, enable_upper_padding, enable_lower_padding, global_offset, K, B, O, Ochannels, apply_relu, CPI, CPO, kernel_id);
  } else {
    // several kernels available, let's split the operation in frames
    int num_rows_kernel[16];
    int enable_upper_padding_kernel[16];
    int enable_lower_padding_kernel[16];
    int global_offset_kernel[16];

    #pragma omp parallel for
    for (int k=0; k<num_kernels; k++) {
      num_rows_kernel[k] = Irows / num_kernels;
      enable_upper_padding_kernel[k] = (k == 0);
      enable_lower_padding_kernel[k] = (k == num_kernels-1);
      global_offset_kernel[k] = Icols * (Irows/num_kernels) * k;
      fpga_conv2D_v1X_kernel(I, Irows, Icols, Ichannels, num_rows_kernel[k], enable_upper_padding_kernel[k], 
		         enable_lower_padding_kernel[k], global_offset_kernel[k], K, B, O, Ochannels, apply_relu, CPI, CPO, k);
    }
  }
}

// fpga_conv2D_v1X: Implementation of conv2D kernels (version 1)
int fpga_conv2D_v1X(ConvolDescriptor *D) {

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

  // This family of kernels need strides of 1x1, kernels of 1x1, padding of 1x2, and batch size 1
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, k_conv2d_cpi, k_conv2d_cpo);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }
    // in case this conv performs also RELU we change the output tensor
    if (D->fpga_apply_relu) O = *(cl::Buffer*)D->fpga_relu_ptrO;
    fpga_conv2D_v1X_launch(I, Irows, Icols, Ichannels, K, B, O, Ochannels, D->fpga_apply_relu, k_conv2d_cpi, k_conv2d_cpo, k_conv2d_num_kernels, k_conv2d_max_rows);
    _profile_fpga_tensor(D->O);
    return 1;
  }

  return 0;
}

#endif
