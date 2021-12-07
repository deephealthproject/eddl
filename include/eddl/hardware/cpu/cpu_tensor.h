/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_CPU_TENSOR_H
#define EDDL_CPU_TENSOR_H

#include "cpu_profile.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

// CPU: Core (static)
void cpu_transpose(Tensor *A, Tensor *B);
void cpu_copy(Tensor *A, Tensor *B);

void cpu_fill_(Tensor *A, float v);
void cpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);

void _cpu_sort(Tensor *A, Tensor *B, bool descending, bool stable);
void cpu_sort(Tensor *A, Tensor *B, bool descending, bool stable);
void cpu_argsort(Tensor *A, Tensor *B, bool descending, bool stable);

void cpu_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void cpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void cpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_gather(Tensor *A, Tensor *B, GatherDescriptor *sd);
void cpu_expand(Tensor *A, Tensor *B, ExpandDescriptor *sd);

void cpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,bool mask_zeros=false); // TODO: Legacy
void cpu_deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc=0,bool mask_zeros=false); // TODO: Legacy

void cpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative);

void cpu_repeat(Tensor* A, Tensor *B, const vector<unsigned int>& repeats, unsigned int axis, bool derivative);

// CPU: Create
void cpu_range(Tensor *A, float min, float step);
void cpu_eye(Tensor *A, int offset);

void cpu_diag(Tensor *A, Tensor *B, int k);

// CPU: Generator
void cpu_rand_uniform(Tensor *A, float v);
void cpu_rand_signed_uniform(Tensor *A, float v);
void cpu_rand_binary(Tensor *A, float v);
void cpu_rand_normal(Tensor *A, float m, float s, bool fast_math);  // TODO: Don't like it

// CPU: Data transformations (2D Optimized) ********************************************
void cpu_shift(Tensor *A, Tensor *B, vector<int> shift, int wrapping_mode, float constant);
void cpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int wrapping_mode, float constant);
void cpu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode);
void cpu_scale_back(Tensor *A, Tensor *B, vector<int> new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode);
void cpu_flip(Tensor *A, Tensor *B, int axis);
void cpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse);
void cpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int wrapping_mode, float constant);
void cpu_pad(Tensor *A, Tensor *B, vector<int> pads);
void cpu_pad_back(Tensor *A, Tensor *B, vector<int> pads);

// CPU: Data transformations (3D Optimized) ********************************************
void cpu_scale3d(Tensor *A, Tensor *B, vector<int> new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode);
void cpu_scale3d_back(Tensor *A, Tensor *B, vector<int> new_shape, int wrapping_mode, float constant, int coordinate_transformation_mode);

// CPU: Data augmentations (2D Optimized) ********************************************
void cpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int wrapping_mode, float constant);
void cpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int wrapping_mode, float constant);
void cpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int wrapping_mode, float constant, int coordinate_transformation_mode);
void cpu_flip_random(Tensor *A, Tensor *B, int axis);
void cpu_crop_random(Tensor *A, Tensor *B);
void cpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int wrapping_mode, float constant);
void cpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);

// CPU: Math (in-place)
void cpu_abs(Tensor *A, Tensor *B);
void cpu_acos(Tensor *A, Tensor *B);
void cpu_add(Tensor *A, Tensor *B, float v);
void cpu_asin(Tensor *A, Tensor *B);
void cpu_atan(Tensor *A, Tensor *B);
void cpu_ceil(Tensor *A, Tensor *B);
void cpu_clamp(Tensor *A, Tensor *B, float min, float max);
void cpu_d_clamp(Tensor *D, Tensor *I, Tensor *PD, float min, float max);
void cpu_cos(Tensor *A, Tensor *B);
void cpu_cosh(Tensor *A, Tensor *B);
void cpu_exp(Tensor *A, Tensor *B);
void cpu_inv(Tensor *A, Tensor *B, float v);
void cpu_floor(Tensor *A, Tensor *B);
void cpu_log(Tensor *A, Tensor *B);
void cpu_log2(Tensor *A, Tensor *B);
void cpu_log10(Tensor *A, Tensor *B);
void cpu_logn(Tensor *A, Tensor *B, float n);
void cpu_mod(Tensor *A, Tensor *B, float v);
void cpu_mult(Tensor *A, Tensor *B, float v);
void cpu_normalize(Tensor *A, Tensor *B, float min, float max);
void cpu_pow(Tensor *A, Tensor *B, float exp);
void cpu_powb(Tensor *A, Tensor *B, float base);
void cpu_remainder(Tensor *A, Tensor *B, float v);
void cpu_round(Tensor *A, Tensor *B);
void cpu_rsqrt(Tensor *A, Tensor *B);
void cpu_sigmoid(Tensor *A, Tensor *B);
void cpu_sign(Tensor *A, Tensor *B, float zero_sign=0.0f);
void cpu_sin(Tensor *A, Tensor *B);
void cpu_sinh(Tensor *A, Tensor *B);
void cpu_sqr(Tensor *A, Tensor *B);
void cpu_sqrt(Tensor *A, Tensor *B);
void cpu_tan(Tensor *A, Tensor *B);
void cpu_tanh(Tensor *A, Tensor *B);
void cpu_trunc(Tensor *A, Tensor *B);

// CPU: Math (static)
void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
void cpu_inc(Tensor *A, Tensor *B);
void cpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
void cpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
void cpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
void cpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void cpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

void cpu_maximum(Tensor* A, Tensor* B, float v);
void cpu_maximum(Tensor* A, Tensor* B, Tensor* C);
void cpu_minimum(Tensor* A, Tensor* B, float v);
void cpu_minimum(Tensor* A, Tensor* B, Tensor* C);

// CPU: Math (reductions)
float cpu_max(Tensor *A);
void cpu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
int cpu_argmax(Tensor *A);
void cpu_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
void cpu_argmax_d(Tensor *D, Tensor *O, Tensor *PD);
std::tuple<float, int> cpu_max(float *ptr, int size, int *map);

float cpu_min(Tensor *A);
void cpu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
int cpu_argmin(Tensor *A);
void cpu_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
std::tuple<float, int> cpu_min(float *ptr, int size, int *map);

float cpu_sum(Tensor *A);
void cpu_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_sum(float *ptr, int size, int *map);

float cpu_sum_abs(Tensor *A);
void cpu_sum_abs(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_sum_abs(float *ptr, int size, int *map);

float cpu_prod(Tensor *A);
void cpu_prod(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_prod(float *ptr, int size, int *map);

float cpu_mean(Tensor *A);
void cpu_mean(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float cpu_median(Tensor *A);
void cpu_median(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
float cpu_median(float *ptr, int size, int *map);

float cpu_var(Tensor *A, bool unbiased);
void cpu_var(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased);
float cpu_var(float *ptr, int size, int *map, bool unbiased);

float cpu_std(Tensor *A, bool unbiased);
void cpu_std(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased);

int cpu_mode(Tensor *A);
void cpu_mode(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);
int cpu_mode(float *ptr, int size, int *map);



// CPU: Reduction
void cpu_reduce(Tensor *A, Tensor *B,string mode,int* map);
void cpu_reduce_op(Tensor *A, Tensor *B,string op,int* map);
void cpu_reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD);
void cpu_reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD);

void cpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
void cpu_reduction(ReduceDescriptor *RD);
void cpu_reduction_back(ReduceDescriptor *RD);
//void cpu_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void cpu_delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void cpu_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
//void cpu_delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);

// CPU: Linear algebra
float cpu_norm(Tensor *A, string ord);
void cpu_norm(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, string ord);
float cpu_norm_(float *ptr, int size, int *map, string ord);


// CPU: Logic functions: Truth value testing
std::pair<unsigned int*, int> cpu_nonzero(Tensor *A);
void cpu_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C);
void cpu_where_back(Tensor *condition, Tensor *PD_A, Tensor *PD_B, Tensor *D);

// CPU: Logic functions: Truth value testing
bool cpu_all(Tensor *A);
bool cpu_any(Tensor *A);

// CPU: Logic functions: Comparisons
void cpu_isfinite(Tensor *A, Tensor* B);
void cpu_isinf(Tensor *A, Tensor* B);
void cpu_isnan(Tensor *A, Tensor* B);
void cpu_isneginf(Tensor *A, Tensor* B);
void cpu_isposinf(Tensor *A, Tensor* B);

// CPU: Logic functions: Comparisons
void cpu_logical_and(Tensor *A, Tensor *B, Tensor *C);
void cpu_logical_or(Tensor *A, Tensor *B, Tensor *C);
void cpu_logical_not(Tensor *A, Tensor *B);
void cpu_logical_xor(Tensor *A, Tensor *B, Tensor *C);

// CPU: Logic functions: Comparisons
bool cpu_allclose_verbose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan);
bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan);
void cpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan);

void cpu_greater(Tensor *A, Tensor *B, float v);

void cpu_greater(Tensor *A, Tensor *B, Tensor *C);
void cpu_greater_equal(Tensor *A, Tensor *B, float v);

void cpu_greater_equal(Tensor *A, Tensor *B, Tensor *C);
void cpu_less(Tensor *A, Tensor *B, float v);

void cpu_less(Tensor *A, Tensor *B, Tensor *C);
void cpu_less_equal(Tensor *A, Tensor *B, float v);

void cpu_less_equal(Tensor *A, Tensor *B, Tensor *C);
void cpu_equal(Tensor *A, Tensor *B, float v);

void cpu_equal(Tensor *A, Tensor *B, Tensor *C);
void cpu_not_equal(Tensor *A, Tensor *B, float v);

void cpu_not_equal(Tensor *A, Tensor *B, Tensor *C);



#endif //EDDL_CPU_TENSOR_H
