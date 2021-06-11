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
// conv kernels
//
//
    
// fpga_conv_kernel: This function launches the kernel execution on the FPGA
void fpga_conv_kernel(cl::Buffer I, cl::Buffer I_add, int H, int W, int rows, int Ichannels, int Ochannels,
                     int first_o_iter, int last_o_iter, int enable_relu, int enable_stm, cl::Buffer K, cl::Buffer B,
                     cl::Buffer O, int global_offset, int enable_upper_padding, int enable_lower_padding,
			         int enable_maxp, int enable_avgp, int enable_clipping, int enable_shift, 
			         int enable_add, int min_clip, int max_clip, int dir_shift, int pos_shift, int CPI, int CPO, int kernel_id) {
  
  // Error variable
  cl_int err;

  // set kernel arguments
  int arg = 0;

  // input iterations
  int I_ITER = (Ichannels + (CPI-1)) / CPI;

  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, I_add));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, H));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, W));
  OCL_CHECK(err, err = kernel_conv2D[kernel_id].setArg(arg++, rows)); 
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
}

// fpga_conv_launch: This function launches all the possible conv2D kernels
void fpga_conv_launch(cl::Buffer I, cl::Buffer I_add, int H, int W, int rows, int Ichannels, int Ochannels,
                     int enable_relu, int enable_stm, cl::Buffer K, cl::Buffer B,
                     cl::Buffer O, int global_offset, int enable_upper_padding, int enable_lower_padding,
			         int enable_maxp, int enable_avgp, int enable_clipping, int enable_shift, 
			         int enable_add, int min_clip, int max_clip, int dir_shift, int pos_shift, int CPI, int CPO, int num_kernels, int max_rows) {
  // Depending on the number of kernels available we split the convolution operation into multiple frames, and launch one thread per kernel
  if (num_kernels == 1) {
    // just one kernel which handles all the conv operation
    int first_o_iter = 0;
    int last_o_iter = ((Ochannels + (CPO-1)) / CPO) - 1;
    fpga_conv_kernel(I, I_add, H, W, rows, Ichannels, Ochannels, first_o_iter, last_o_iter, enable_relu,
        enable_stm, K, B, O, global_offset, enable_upper_padding, enable_lower_padding, enable_maxp,
		enable_avgp, enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift, CPI, CPO, 0);
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
      fpga_conv_kernel(I, I_add, H, W, rows, Ichannels, Ochannels, first_o_iter, last_o_iter, enable_relu,
              enable_stm, K, B, O, global_offset, enable_upper_padding, enable_lower_padding, enable_maxp,
		          enable_avgp, enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift, CPI, CPO, k);    
    }
  }
}

//Transform de ConvolDescriptor in OpenCL variables
int fpga_k_conv(ConvolDescriptor *D, Tensor *ADD, int enable_relu, int enable_stm, int global_offset, 
                     int enable_upper_padding, int enable_lower_padding, int enable_maxp, 
                     int enable_avgp, int enable_clipping, int enable_shift, int enable_add, 
                     int min_clip, int max_clip, int dir_shift, int pos_shift) {
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
  cl::Buffer I_add;                                   // input add data

  // This family of kernels need strides of 1x1, kernels of 1x1, padding of 1x1, and batch size 1
  if ((stride_rows == 1) && (stride_cols == 1) && (Krows == 3) && (Kcols == 3) && (batch_size == 1) && (padding_rows == 1) && (padding_cols == 1)) {
    
    if(enable_add)  I_add = *(cl::Buffer*) ADD->fpga_ptr;
    else I_add = *(cl::Buffer*) fpga_create_memory(sizeof(float)); // Creating dummy buffer for add buffer
    fpga_conv_launch(I, I_add, Irows, Icols, Irows, Ichannels, Ochannels, enable_relu, enable_stm, K,
          B, O, global_offset, enable_upper_padding, enable_lower_padding, enable_maxp, enable_avgp, 
          enable_clipping, enable_shift, enable_add, min_clip, max_clip, dir_shift, pos_shift, k_conv2d_cpi,
          k_conv2d_cpo, k_conv2d_num_kernels, k_conv2d_max_rows);

    return 1;
  }
printf("(stride_rows == %d (1)) && (stride_cols == %d (1)) && (Krows == %d (3)) && (Kcols == %d (3)) && (batch_size == %d (1)) && (batch_size == %d (1)) && (padding_cols == %d (1)\n",
stride_rows, stride_cols,Krows,Kcols,batch_size,batch_size, padding_cols);
  return 0;
}
#endif
