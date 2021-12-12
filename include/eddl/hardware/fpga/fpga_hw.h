/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#ifndef EDDL_FPGA_HW_H
#define EDDL_FPGA_HW_H

// --------------------------------------------------------------------------------------------------------
#include "fpga_profile.h"
#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"
#include <ap_fixed.h>                       // Aproximated precision fixed point support
#include <ap_int.h>                         // Aproximated precision integer support

#include "eddl/hardware/fpga/xcl2.hpp"

extern cl::CommandQueue *q;

//#define FPGA_DEBUG

#include "eddl/hardware/fpga/fpga_enables.h"

// ----------------------------------------------------------------------------------------------------------
// Precision support
//
// data_type is the basic precision format used for tensors in FPGA
//
// Three supported formats:
//  - float. This format has been tested and works well within EDDL. Indeed, is the single format used by EDDL
//  - ap_fixed. This format is under test. Aproximated precision fixed point format
//  - ap_int. This format is under test. Aproximated precision integer
//
// The PRECISION_CONVERSION define must be set if either ap_fixed or ap_int formats are used. This define
// enables the precision conversion from CPU to FPGA and viceversa. Whenever a tensor is read or written
// from/to the FPGA the precision conversion is performed on a temporary CPU buffer.
// If float precision is used, then PRECISION_CONVERSION should not be used
//
//#define PRECISION_CONVERSION
//#define MIN_FLOAT_PRECISION_CONVERSION -32767
//#define MAX_FLOAT_PRECISION_CONVERSION 32768
#define fpga_data_type float
//#define fpga_data_type ap_fixed<8,4,AP_TRN,AP_WRAP>
//#define fpga_data_type ap_fixed<16,8,AP_TRN,AP_WRAP>
//#define fpga_data_type ap_int<8>
//#define fpga_data_type ap_int<16>
//#define fpga_data_type ap_int<32>
//
//


#define MAX_KERNELS 16

// Debug functions
void _debug_fpga_funcs(const char *str);


// conv kernels (3)
extern cl::Kernel kernel_hlsinf[16];

// conv2d kernel related global variables

#define HLSINF_FP32  0
#define HLSINF_API8  1
#define HLSINF_APUI8 2 
#define HLSINF_API32 3

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

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

void set_callback(cl::Event event, const char *queue_name);
void event_cb(cl_event event1, cl_int cmd_status, void *data);

void fpga_init(int kernel_version, int kernel_subversion);

void fpga_destroy_memory(cl::Buffer *fpga_ptrI);
cl::Buffer *fpga_create_memory(long int size);
void fpga_copy_memory_to_fpga(void *ptr_cpu, cl::Buffer *ptr_fpga, long int size);
void fpga_copy_memory_to_fpga_and_format(void *ptr_cpu, cl::Buffer *ptr_fpga, long int size, int src_format, int dst_format);
void fpga_copy_memory_from_fpga(cl::Buffer *ptr_fpga, void *ptr_cpu, long int size);

void fpga_copy_fpga(Tensor *A, Tensor *B);
void fpga_copy_to_fpga_good(float *nptr, Tensor *A, int cvt=1);
void fpga_copy_from_fpga_good(Tensor *A,float *nptr, int cvt=1);
void fpga_copy_from_fpga(Tensor *A,float *nptr, int cvt=1);



void fpga_transform_nn(Tensor *A, Tensor *B, int copy_cpu_to_fpga, int copy_fpga_to_cpu, int transform);

void filter_IHW_to_GIHWCPI(Tensor *A, Tensor *B);
void dense_to_conv(float *ptr_src, int N, int M, float *ptr_dst, int I, int O, int KH, int KW);
void tensor_padded(Tensor *A, Tensor *B);
void get_batch_norm_values(int ochannels, Tensor *global_mean, Tensor *global_variance, Tensor* affine_g, Tensor* affine_b, Tensor* output); 
void fpga_write_buffer(char *file_name, void *ptr, int size, int data_size);

#endif //EDDL_FPGA_HW_H

#endif
