#ifdef cFPGA

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/profiling.h"

extern vector<cl::Event> kernel_events; // Kernel events (completion)

PROFILING_ENABLE_EXTERN(fpga_Conv2D_RELU_MAXPOOL);
// -----------------------------------------------------------------
// Conv2D + ReLU + Maxpool of kernels
//
//
//TODO
// fpga_conv_relu_maxpool_kernel: This function launches the kernel execution on the FPGA
void fpga_conv_relu_maxpool_kernel(cl::Buffer I, int Irows, int Icols, int Ichannels, 
			    cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels, int CPI, int CPO, 
			    int first_o_iter, int last_o_iter, int kernel_id) {

  PROFILING_HEADER(fpga_Conv2D_RELU_MAXPOOL);

  int enable_relu = 1;
  int enable_stm = 0;
  int H = Irows;                // input channel height
  int W = Icols;                // input channel width
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

  // Error variable
  cl_int err;
  
  // input iterations
  int I_ITER = (Ichannels + (CPI-1)) / CPI;

  // set kernel arguments
  int arg = 0;

  cl::Buffer d_buffer;
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I)); 
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Irows));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Icols));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Irows)); 
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Ichannels));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, Ochannels));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I_ITER));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, first_o_iter));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, last_o_iter));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_relu));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_stm));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, K));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, B));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, O));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, global_offset));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_upper_padding));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, enable_lower_padding));
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
  
  PROFILING_FOOTER(fpga_Conv2D_RELU_MAXPOOL);
}

// fgpa_conv2D_v2X_launch: This function launches all the possible conv2D kernels
void fpga_conv_relu_maxpool_launch(cl::Buffer I, int Irows, int Icols, int Ichannels, 
		            cl::Buffer K, cl::Buffer B, cl::Buffer O, int Ochannels, 
			    int CPI, int CPO, int num_kernels, int max_rows) {

  // Depending on the number of kernels available we split the convolution operation into multiple frames, and launch one thread per kernel
  if (num_kernels == 1) {
    // just one kernel which handles all the conv operation
    int first_o_iter = 0;
    int last_o_iter = ((Ochannels + (CPO-1)) / CPO) - 1;
    fpga_conv_relu_maxpool_kernel(I, Irows, Icols, Ichannels, K, B, O, Ochannels, CPI, CPO, first_o_iter, last_o_iter, 0);
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
      fpga_conv_relu_maxpool_kernel(I, Irows, Icols, Ichannels, K, B, O, Ochannels, CPI, CPO, first_o_iter, last_o_iter,  k);
    }
  }
}

//Transform de ConvolDescriptor in OpenCL variables
void fpga_conv_relu_maxpool_transform(ConvolDescriptor *D) {
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

  // This family of kernels need strides of 1x1, kernels of 1x1, padding of 1x1, and batch size 1
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    // This kernel needs the data kernel in the format GO x GI x CPO x CPI x KH x KW
    // If not converted yet then we do it now
    if (!D->fpga_kernel_in_fpga_format) {
      fpga_reshape_kernel_data_convol(D, 3, 3, Ichannels, Ochannels, k_conv2d_cpi, k_conv2d_cpo);
      D->fpga_kernel_in_fpga_format = 1;
      K     = *(cl::Buffer*)D->K->fpga_ptr; // read again the pointer since it may be changed
    }
    int enable_maxp = 0;
    fpga_conv_relu_maxpool_launch(I, Irows, Icols, Ichannels, K, B, O, Ochannels, k_conv2d_cpi, k_conv2d_cpo, k_conv2d_num_kernels, k_conv2d_max_rows);
    _profile_fpga_tensor(D->O);
  }
}
#endif