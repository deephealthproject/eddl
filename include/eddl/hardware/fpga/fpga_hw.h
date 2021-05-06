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

#include "fpga_profile.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"
#include <ap_fixed.h>                       // Aproximated precision fixed point support
#include <ap_int.h>                         // Aproximated precision integer support


extern cl::CommandQueue *q;

//#define FPGA_DEBUG
//#define FPGA_DEBUG_TENSORS
//#define FPGA_DEBUG_FUNCS
//#define FPGA_DEBUG_VERBOSE

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
//#define fpga_data_type float
#define fpga_data_type ap_fixed<8,4,AP_TRN,AP_WRAP>
//#define fpga_data_type ap_int<8>

// Debug functions
void _debug_fpga_funcs(const char *str);


// activation kernels (24)
extern cl::Kernel kernel_relu,   kernel_d_relu,  kernel_thresholded_relu,    kernel_d_thresholded_relu, kernel_leaky_relu,     kernel_d_leaky_relu;
extern cl::Kernel kernel_elu,    kernel_d_elu,   kernel_softplus,            kernel_d_softplus,         kernel_softsign,       kernel_d_softsign;
extern cl::Kernel kernel_linear, kernel_d_linear,kernel_sigmoid,             kernel_d_sigmoid,          kernel_hard_sigmoid,   kernel_d_hard_sigmoid;
extern cl::Kernel kernel_exp,    kernel_d_exp,   kernel_tanh,                kernel_d_tanh,             kernel_softmax,        kernel_d_softmax;

// bn kernels (4)
extern cl::Kernel kernel_permute_channels_last, kernel_permute_channels_first;
extern cl::Kernel kernel_permute_batch_last,    kernel_permute_batch_first;

// comparison kernels (20)
extern cl::Kernel kernel_all,         kernel_any,        kernel_isfinite,    kernel_isinf;
extern cl::Kernel kernel_isnan,       kernel_isneginf,   kernel_isposinf,    kernel_equal2;
extern cl::Kernel kernel_logical_and, kernel_logical_or, kernel_logical_not, kernel_logical_xor;
extern cl::Kernel kernel_allclose,    kernel_isclose,    kernel_greater,     kernel_greater_equal;
extern cl::Kernel kernel_less,        kernel_less_equal, kernel_equal,       kernel_not_equal;
extern cl::Kernel kernel_greater_vector, kernel_greater_equal_vector, kernel_less_vector;
extern cl::Kernel kernel_less_equal_vector, kernel_equal_vector, kernel_not_equal_vector;

// core kernels (11)
extern cl::Kernel kernel_transpose,   kernel_copy,        kernel_fill_,      kernel_fill;
extern cl::Kernel kernel_select,      kernel_select_back, kernel_set_select, kernel_set_select_back;
extern cl::Kernel kernel_set_select2, kernel_deselect,    kernel_concat;
extern cl::Kernel kernel_select_nn,   kernel_select_back_nn, kernel_set_select_back_nn, kernel_set_select_nn;

// conv kernels (3)
extern cl::Kernel kernel_im2col;
extern cl::Kernel kernel_conv2D[16];

// create kernels (3)
extern cl::Kernel kernel_range, kernel_eye, kernel_diag;

// da kernels (6)
extern cl::Kernel kernel_single_shift, kernel_single_rotate, kernel_single_scale;
extern cl::Kernel kernel_single_flip,  kernel_single_crop;
extern cl::Kernel kernel_crop_scale_random;

// generator kernels (4)
extern cl::Kernel kernel_rand_uniform, kernel_signed_uniform, kernel_rand_binary, kernel_rand_normal;

// losses kernels (1)
extern cl::Kernel kernel_cent;

// metrics kernels (2)
extern cl::Kernel kernel_accuracy, kernel_bin_accuracy;

// pool kernels (4)
extern cl::Kernel kernel_mpool2D, kernel_mpool2D_back, kernel_avgpool2D, kernel_avgpool2D_back;

// reduction kernels (5)
extern cl::Kernel kernel_reduce, kernel_reduce_op, kernel_reduce_sum2D, kernel_reduction, kernel_reduction_back;

// tensor_nn kernels (2)
extern cl::Kernel kernel_repeat_nn, kernel_d_repeat_nn;

// math kernels (46)
extern cl::Kernel kernel_abs,       kernel_acos,   kernel_add,      kernel_asin,       kernel_atan,          kernel_ceil,          kernel_clamp;
extern cl::Kernel kernel_cos,       kernel_cosh,   kernel_mod,      kernel_mult,       kernel_trunc,         kernel_sum_abs;
extern cl::Kernel kernel_floor,     kernel_inv,    kernel_log,      kernel_log2,       kernel_log10,         kernel_logn;
extern cl::Kernel kernel_normalize, kernel_pow,    kernel_powb,     kernel_reciprocal, kernel_remainder,     kernel_round,         kernel_rsqrt;
extern cl::Kernel kernel_sign,      kernel_sin,    kernel_sinh,     kernel_sqr,        kernel_sqrt,          kernel_tan;
extern cl::Kernel kernel_inc,       kernel_el_div, kernel_el_mult,  kernel_sign2,      kernel_sum2D_rowwise, kernel_sum2D_colwise;
extern cl::Kernel kernel_max,       kernel_min,    kernel_sum,      kernel_mult2d;

// conv2d kernel related global variables
#ifdef K_ENABLED_CONV2D
extern int k_conv2d_cpi;
extern int k_conv2d_cpo;
extern int k_conv2d_num_kernels;
extern int k_conv2d_max_rows; 
#endif

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

void set_callback(cl::Event event, const char *queue_name);
void event_cb(cl_event event1, cl_int cmd_status, void *data);

void fpga_init();
cl::Buffer *fpga_create_tensor(int device, int size);
void fpga_delete_tensor(int device, cl::Buffer *ptr, int fpga_tensor_id_p, int size);

void fpga_destroy_memory(cl::Buffer *fpga_ptrI);
cl::Buffer *fpga_create_memory(long int size);
void fpga_copy_memory_to_fpga(void *ptr_cpu, cl::Buffer *ptr_fpga, long int size);
void fpga_copy_memory_from_fpga(cl::Buffer *ptr_fpga, void *ptr_cpu, long int size);

void fpga_copy_fpga(Tensor *A, Tensor *B);
void fpga_copy_to_fpga(float *nptr, Tensor *A, int cvt=1);
void fpga_copy_from_fpga(Tensor *A,float *nptr, int cvt=1);
void fpga_copy_addresses_from_fpga(SelDescriptor *SD, int size, int *nptr);


// CPU: Core (static)
void fpga_transpose(Tensor *A, Tensor *B);
void fpga_copy(Tensor *A, Tensor *B);

void fpga_fill_(Tensor *A, float v);
void fpga_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);

void fpga_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void fpga_set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void fpga_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,bool mask_zeros=false);
void fpga_deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc=0,bool mask_zeros=false);

void fpga_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative);

void fpga_repeat(Tensor *A, Tensor *B, vector<int> size);
void fpga_d_repeat(Tensor *D, Tensor *A, vector<int> size);

void fpga_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void fpga_set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

// CPU: Create
void fpga_range(Tensor *A, float min, float step);
void fpga_eye(Tensor *A, int offset);
void fpga_diag(Tensor *A, Tensor *B, int k);

// CPU: Generator
void fpga_rand_uniform(Tensor *A, float v);
void fpga_rand_signed_uniform(Tensor *A, float v);
void fpga_rand_binary(Tensor *A, float v);
void fpga_rand_normal(Tensor *A, float m, float s, bool fast_math);  // TODO: Don't like it

// CPU: Data transformations (2D Optimized) ********************************************
void fpga_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant);
void fpga_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant);
void fpga_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant);
void fpga_flip(Tensor *A, Tensor *B, int axis);
void fpga_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse);
void fpga_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant);

// CPU: Data augmentations (2D Optimized) ********************************************
void fpga_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant);
void fpga_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant);
void fpga_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void fpga_flip_random(Tensor *A, Tensor *B, int axis);
void fpga_crop_random(Tensor *A, Tensor *B);
void fpga_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void fpga_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);

// FPGA: Math (in-place)
void fpga_abs(Tensor *A, Tensor *B);
void fpga_acos(Tensor *A, Tensor *B);
void fpga_add(Tensor *A, Tensor *B, float v);
void fpga_asin(Tensor *A, Tensor *B);
void fpga_atan(Tensor *A, Tensor *B);
void fpga_ceil(Tensor *A, Tensor *B);
void fpga_clamp(Tensor *A, Tensor *B, float min, float max);
void fpga_cos(Tensor *A, Tensor *B);
void fpga_cosh(Tensor *A, Tensor *B);
void fpga_exp(Tensor *A, Tensor *B);
void fpga_inv(Tensor *A, Tensor *B, float v);
void fpga_floor(Tensor *A, Tensor *B);
void fpga_log(Tensor *A, Tensor *B);
void fpga_log2(Tensor *A, Tensor *B);
void fpga_log10(Tensor *A, Tensor *B);
void fpga_logn(Tensor *A, Tensor *B, float n);
void fpga_mod(Tensor *A, Tensor *B, float v);
void fpga_mult(Tensor *A, Tensor *B, float v);
void fpga_normalize(Tensor *A, Tensor *B, float min, float max);
void fpga_pow(Tensor *A, Tensor *B, float exp);
void fpga_powb(Tensor *A, Tensor *B, float base);
void fpga_remainder(Tensor *A, Tensor *B, float v);
void fpga_round(Tensor *A, Tensor *B);
void fpga_rsqrt(Tensor *A, Tensor *B);
void fpga_sigmoid(Tensor *A, Tensor *B);
void fpga_sign(Tensor *A, Tensor *B, float zero_sign=0.0f);
void fpga_sin(Tensor *A, Tensor *B);
void fpga_sinh(Tensor *A, Tensor *B);
void fpga_sqr(Tensor *A, Tensor *B);
void fpga_sqrt(Tensor *A, Tensor *B);
void fpga_tan(Tensor *A, Tensor *B);
void fpga_tanh(Tensor *A, Tensor *B);
void fpga_trunc(Tensor *A, Tensor *B);

// CPU: Math (static)
void fpga_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
void fpga_inc(Tensor *A, Tensor *B);
void fpga_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
void fpga_el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
void fpga_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void fpga_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

void fpga_maximum(Tensor* A, Tensor* B, float v);
void fpga_maximum(Tensor* A, Tensor* B, Tensor* C);
void fpga_minimum(Tensor* A, Tensor* B, float v);
void fpga_minimum(Tensor* A, Tensor* B, Tensor* C);

// CPU: Math (reductions)
float fpga_max(Tensor *A);
void fpga_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
int fpga_argmax(Tensor *A);
void fpga_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
std::tuple<float, int> fpga_max(float *ptr, int size, int *map);

// FPGA: Logic functions: Truth value testing
void fpga_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C);

float fpga_min(Tensor *A);
void fpga_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
int fpga_argmin(Tensor *A);
void fpga_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
std::tuple<float, int> fpga_min(float *ptr, int size, int *map);

float fpga_sum(Tensor *A);
void fpga_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float fpga_sum(float *ptr, int size, int *map);

float fpga_sum_abs(Tensor *A);
void fpga_sum_abs(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float fpga_sum_abs(float *ptr, int size, int *map);

float fpga_prod(Tensor *A);
void fpga_prod(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float fpga_prod(float *ptr, int size, int *map);

float fpga_mean(Tensor *A);
void fpga_mean(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float fpga_median(Tensor *A);
void fpga_median(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float fpga_median(float *ptr, int size, int *map);

float fpga_var(Tensor *A, bool unbiased);
void fpga_var(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased);
float fpga_var(float *ptr, int size, int *map, bool unbiased);

float fpga_std(Tensor *A, bool unbiased);
void fpga_std(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased);

int fpga_mode(Tensor *A);
void fpga_mode(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
int fpga_mode(float *ptr, int size, int *map);

// CPU: Should be reductions
float fpga_max(Tensor *A);
float fpga_min(Tensor *A);
float fpga_sum(Tensor *A);
float fpga_sum_abs(Tensor *A);

// CPU: Reduction
void fpga_reduce(Tensor *A, Tensor *B,string mode,int* map);
void fpga_reduce_op(Tensor *A, Tensor *B,string op,int* map);
void fpga_reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD);
void fpga_reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD);

void fpga_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
void fpga_reduction(ReduceDescriptor *RD);
void fpga_reduction_back(ReduceDescriptor *RD);
//void fpga_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void fpga_delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void fpga_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
//void fpga_delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);

// CPU: Logic functions: Truth value testing
bool fpga_all(Tensor *A);
bool fpga_any(Tensor *A);

// CPU: Logic functions: Comparisons
void fpga_isfinite(Tensor *A, Tensor* B);
void fpga_isinf(Tensor *A, Tensor* B);
void fpga_isnan(Tensor *A, Tensor* B);
void fpga_isneginf(Tensor *A, Tensor* B);
void fpga_isposinf(Tensor *A, Tensor* B);

// CPU: Logic functions: Comparisons
void fpga_logical_and(Tensor *A, Tensor *B, Tensor *C);
void fpga_logical_or(Tensor *A, Tensor *B, Tensor *C);
void fpga_logical_not(Tensor *A, Tensor *B);
void fpga_logical_xor(Tensor *A, Tensor *B, Tensor *C);

// CPU: Logic functions: Comparisons
bool fpga_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan);
void fpga_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan);
void fpga_greater(Tensor *A, Tensor *B, Tensor *C);
void fpga_greater_equal(Tensor *A, Tensor *B, Tensor *C);
void fpga_less(Tensor *A, Tensor *B, Tensor *C);
void fpga_less_equal(Tensor *A, Tensor *B, Tensor *C);
void fpga_equal(Tensor *A, Tensor *B, Tensor *C);
void fpga_not_equal(Tensor *A, Tensor *B, Tensor *C);
void fpga_greater(Tensor *A, Tensor *B, float v);
void fpga_greater_equal(Tensor *A, Tensor *B, float v);
void fpga_less(Tensor *A, Tensor *B, float v);
void fpga_less_equal(Tensor *A, Tensor *B, float v);
void fpga_equal(Tensor *A, Tensor *B, float v);
void fpga_not_equal(Tensor *A, Tensor *B, float v);

// Legacy
int fpga_equal2(Tensor *A, Tensor *B, float epsilon);



#endif //EDDL_FPGA_HW_H

#endif
