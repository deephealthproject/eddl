/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#ifndef EDDL_FPGA_HW_H
#define EDDL_FPGA_HW_H

// openCL headers, vendor specific 
#ifdef cFPGA_VENDOR_XILINX
  #include "eddl/hardware/fpga/xilinx/fpga_xilinx_hw.h"
#endif

#ifdef cFPGA_VENDOR_INTEL
 #include "eddl/hardware/fpga/intel/fpga_intel_hw.h"
#endif

// --------------------------------------------------------------------------------------------------------
#include "fpga_profile.h"
#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"

//#define FPGA_DEBUG
//#define WRITE_TENSORS_TO_FILE
#define WRITE_TENSORS_TO_FILE_ASCI


#define MAX_KERNELS 16

// Debug functions
void _debug_fpga_funcs(const char *str);



// conv2d kernel related global variables

#define HLSINF_FP32      0
#define HLSINF_API8      1
#define HLSINF_APUI8     2 
#define HLSINF_API32     3
#define HLSINF_APF_8_4   4
#define HLSINF_APF_16_8  5
#define HLSINF_APF_32_16 6

extern int hlsinf_filter_format;
extern int hlsinf_bias_format;
extern int hlsinf_input_format;
extern int hlsinf_output_format;
extern int hlsinf_cpi;
extern int hlsinf_cpo;
extern int hlsinf_num_kernels;
extern int hlsinf_ho_max;
extern int hlsinf_wo_max;
extern int hlsinf_max_rows;
extern std::string hlsinf_xclbin;
extern bool hlsinf_conv_support;
extern bool hlsinf_shift_support;
extern bool hlsinf_clip_support;
extern bool hlsinf_relu_support;
extern bool hlsinf_stm_support;
extern bool hlsinf_maxp_support;
extern bool hlsinf_avgp_support;
extern bool hlsinf_bn_support;
extern bool hlsinf_add_support;
extern bool hlsinf_upsize_support;
extern bool hlsinf_dense_support;
extern int  hlsinf_weight_buffer;
extern int  hlsinf_data_buffer;

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

// Following values match their analogues defined for Intel fpga in /opt/.../include/CL/cl.h, 
#define FPGA_CLMEM_READ_WRITE                           (1 << 0)  // CL_MEM_READ_WRITE
#define FPGA_CLMEM_WRITE_ONLY                           (1 << 1)  // CL_MEM_WRITE_ONLY
#define FPGA_CLMEM_READ_ONLY                            (1 << 2)  // CL_MEM_READ_ONLY
#define FPGA_CLMEM_USE_HOST_PTR                         (1 << 3)  // CL_MEM_USE_HOST_PTR
#define FPGA_CLMEM_ALLOC_HOST_PTR                       (1 << 4)  // CL_MEM_ALLOC_HOST_PTR
#define FPGA_CLMEM_COPY_HOST_PTR                        (1 << 5)  // CL_MEM_COPY_HOST_PTR
/* reserved                                         (1 << 6)    */

// vendor-specific void set_callback(cl::Event event, const char *queue_name);
void event_cb(cl_event event1, cl_int cmd_status, void *data);

void fpga_init(int kernel_version, int kernel_subversion);

//void *fpga_create_memory(unsigned long flags, long int size);
//void fpga_copy_memory_to_fpga(void *ptr_cpu, void *ptr_fpga, long int size);
//void fpga_copy_memory_to_fpga_and_format(void *ptr_cpu, void *ptr_fpga, long int size, int src_format, int dst_format);
//void fpga_copy_memory_from_fpga(void *ptr_fpga, void *ptr_cpu, long int size);
//
//void fpga_transform_nn(Tensor *A, Tensor *B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform);

void filter_IHW_to_GIHWCPI(Tensor *A, Tensor *B);
void dense_to_conv(float *ptr_src, int N, int M, float *ptr_dst, int I, int O, int KH, int KW);
void tensor_padded(Tensor *A, Tensor *B);
void get_batch_norm_values(int ochannels, Tensor *global_mean, Tensor *global_variance, Tensor* affine_g, Tensor* affine_b, Tensor* output); 
void fpga_write_buffer(char *file_name, void *ptr, int size, int data_size);
void fpga_write_buffer(char *file_name, Tensor *ptr, int size, int data_size);

#endif //EDDL_FPGA_HW_H

#endif
