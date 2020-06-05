/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifndef EDDL_FPGA_HW_H
#define EDDL_FPGA_HW_H

#include "fpga_profile.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

void fpga_init();
void fpga_create_tensor(Tensor *T, int dev);
void fpga_copy_to_fpga(float *nptr, Tensor *A);
void fpga_copy_from_fpga(Tensor *A,float *nptr);


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

// CPU: Create
void fpga_range(Tensor *A, float min, float step);
void fpga_eye(Tensor *A, int offset);

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

// CPU: Math (in-place)
void fpga_abs_(Tensor *A);
void fpga_acos_(Tensor *A);
void fpga_add_(Tensor *A, float v);
void fpga_asin_(Tensor *A);
void fpga_atan_(Tensor *A);
void fpga_ceil_(Tensor *A);
void fpga_clamp_(Tensor *A, float min, float max);
void fpga_cos_(Tensor *A);
void fpga_cosh_(Tensor *A);
void fpga_exp_(Tensor *A);
void fpga_inv_(Tensor *A, float v);
void fpga_floor_(Tensor *A);
void fpga_log_(Tensor *A);
void fpga_log2_(Tensor *A);
void fpga_log10_(Tensor *A);
void fpga_logn_(Tensor *A, float n);
void fpga_mod_(Tensor *A, float v);
void fpga_mult_(Tensor *A, float v);
void fpga_normalize_(Tensor *A, float min, float max);
void fpga_pow_(Tensor *A, float exp);
void fpga_powb_(Tensor *A, float base);
void fpga_reciprocal_(Tensor *A);
void fpga_remainder_(Tensor *A, float v);
void fpga_round_(Tensor *A);
void fpga_rsqrt_(Tensor *A);
void fpga_sigmoid_(Tensor *A);
void fpga_sign_(Tensor *A);
void fpga_sin_(Tensor *A);
void fpga_sinh_(Tensor *A);
void fpga_sqr_(Tensor *A);
void fpga_sqrt_(Tensor *A);
void fpga_tan_(Tensor *A);
void fpga_tanh_(Tensor *A);
void fpga_trunc_(Tensor *A);

// CPU: Math (static)
void fpga_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
void fpga_inc(Tensor *A, Tensor *B);
void fpga_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
void fpga_el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
void fpga_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
void fpga_sign2(Tensor *A, Tensor *B); // TODO: Remove
void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void fpga_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

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

// Legacy
int fpga_equal2(Tensor *A, Tensor *B, float epsilon);



#endif //EDDL_FPGA_HW_H
