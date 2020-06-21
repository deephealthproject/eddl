/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_GPU_HW_H
#define EDDL_GPU_HW_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/tensor_reduction.h"
#include "eddl/descriptors/descriptors.h"

#define PRECISION_FLOAT -std::numeric_limits<float>::max()


// CPU: Core (static)
//void gpu_transpose(Tensor *A, Tensor *B);
//void gpu_copy(Tensor *A, Tensor *B);

void gpu_fill_(Tensor *A, float v);
void gpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);

void _gpu_sort(Tensor *A, Tensor *B, bool descending, bool stable);
void gpu_sort(Tensor *A, Tensor *B, bool descending, bool stable);
void gpu_argsort(Tensor *A, Tensor *B, bool descending, bool stable);

void gpu_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void gpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void gpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative);

// GPU: Create
void gpu_range(Tensor *A, float start, float step);
void gpu_eye(Tensor *A, int offset);

void gpu_diag(Tensor *A, Tensor *B, int k);

// GPU: Generator
void gpu_rand_uniform(Tensor *A, float v);
void gpu_rand_signed_uniform(Tensor *A, float v);
void gpu_rand_binary(Tensor *A, float v);
void gpu_rand_normal(Tensor *A, float m, float s);

// GPU: Data transformations (2D Optimized) ********************************************
void gpu_shift(Tensor *A, Tensor *B, vector<int> t_shift, int mode, float constant);
void gpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant);
void gpu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant);
void gpu_flip(Tensor *A, Tensor *B, int axis);
void gpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse);
void gpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant);

// GPU: Data augmentations (2D Optimized) ********************************************
void gpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant);
void gpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant);
void gpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void gpu_flip_random(Tensor *A, Tensor *B, int axis);
void gpu_crop_random(Tensor *A, Tensor *B);
void gpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void gpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);

// CPU: Math (in-place)
void gpu_abs(Tensor *A, Tensor *B);
void gpu_acos(Tensor *A, Tensor *B);
void gpu_add(Tensor *A, Tensor *B, float v);
void gpu_asin(Tensor *A, Tensor *B);
void gpu_atan(Tensor *A, Tensor *B);
void gpu_ceil(Tensor *A, Tensor *B);
void gpu_clamp(Tensor *A, Tensor *B, float min, float max);
void gpu_cos(Tensor *A, Tensor *B);
void gpu_cosh(Tensor *A, Tensor *B);
void gpu_exp(Tensor *A, Tensor *B);
void gpu_inv(Tensor *A, Tensor *B, float v);
void gpu_floor(Tensor *A, Tensor *B);
void gpu_log(Tensor *A, Tensor *B);
void gpu_log2(Tensor *A, Tensor *B);
void gpu_log10(Tensor *A, Tensor *B);
void gpu_logn(Tensor *A, Tensor *B, float n);
void gpu_mod(Tensor *A, Tensor *B, float v);
void gpu_mult(Tensor *A, Tensor *B, float v);
void gpu_normalize(Tensor *A, Tensor *B, float min, float max);
void gpu_pow(Tensor *A, Tensor *B, float exp);
void gpu_powb(Tensor *A, Tensor *B, float base);
void gpu_remainder(Tensor *A, Tensor *B, float v);
void gpu_round(Tensor *A, Tensor *B);
void gpu_rsqrt(Tensor *A, Tensor *B);
void gpu_sigmoid(Tensor *A, Tensor *B);
void gpu_sign(Tensor *A, Tensor *B, float zero_sign=0.0f);
void gpu_sin(Tensor *A, Tensor *B);
void gpu_sinh(Tensor *A, Tensor *B);
void gpu_sqr(Tensor *A, Tensor *B);
void gpu_sqrt(Tensor *A, Tensor *B);
void gpu_tan(Tensor *A, Tensor *B);
void gpu_tanh(Tensor *A, Tensor *B);
void gpu_trunc(Tensor *A, Tensor *B);

// CPU: Math (static)
void gpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
void gpu_inc(Tensor *A, Tensor *B);
void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
void gpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
void gpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

void gpu_maximum(Tensor* A, Tensor* B, float v);
void gpu_maximum(Tensor* A, Tensor* B, Tensor* C);
void gpu_minimum(Tensor* A, Tensor* B, float v);
void gpu_minimum(Tensor* A, Tensor* B, Tensor* C);

// GPU: Should be reductions
float gpu_max(Tensor *A);
void gpu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

int gpu_argmax(Tensor *A);
void gpu_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_min(Tensor *A);
void gpu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

int gpu_argmin(Tensor *A);
void gpu_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_sum(Tensor *A);
void gpu_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_sum_abs(Tensor *A);
void gpu_sum_abs(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_prod(Tensor *A);
void gpu_prod(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_mean(Tensor *A);
void gpu_mean(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_median(Tensor *A);
void gpu_median(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);

float gpu_var(Tensor *A, bool unbiased);
void gpu_var(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased);

float gpu_std(Tensor *A, bool unbiased);
void gpu_std(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased);

int gpu_mode(Tensor *A);
void gpu_mode(Tensor *A, Tensor *B, ReduceDescriptor2 *rd);


// GPU: Reduction
void gpu_reduce(Tensor *A, Tensor *B,string mode,int* map);
void gpu_reduce_op(Tensor *A, Tensor *B,string op,int* map);
void gpu_reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD);
void gpu_reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD);

void gpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
void gpu_reduction(ReduceDescriptor *RD);
void gpu_reduction_back(ReduceDescriptor *RD);
//void gpu_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void gpu_delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void gpu_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
//void gpu_delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);

// GPU: Linear algebra
float gpu_norm(Tensor *A, string ord);
void gpu_norm(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, string ord);

// Generating index arrays *****************************
void gpu_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C);

// GPU: Logic functions: Truth value testing
bool gpu_all(Tensor *A);
bool gpu_any(Tensor *A);

// GPU: Logic functions: Comparisons
void gpu_isfinite(Tensor *A, Tensor* B);
void gpu_isinf(Tensor *A, Tensor* B);
void gpu_isnan(Tensor *A, Tensor* B);
void gpu_isneginf(Tensor *A, Tensor* B);
void gpu_isposinf(Tensor *A, Tensor* B);

// GPU: Logic functions: Comparisons
void gpu_logical_and(Tensor *A, Tensor *B, Tensor *C);
void gpu_logical_or(Tensor *A, Tensor *B, Tensor *C);
void gpu_logical_not(Tensor *A, Tensor *B);
void gpu_logical_xor(Tensor *A, Tensor *B, Tensor *C);

// GPU: Logic operations: Comparison ops
bool gpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan);
void gpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan);

void gpu_greater(Tensor *A, Tensor *B, float v);
void gpu_greater(Tensor *A, Tensor *B, Tensor *C);
void gpu_greater_equal(Tensor *A, Tensor *B, float v);
void gpu_greater_equal(Tensor *A, Tensor *B, Tensor *C);
void gpu_less(Tensor *A, Tensor *B, float v);
void gpu_less(Tensor *A, Tensor *B, Tensor *C);
void gpu_less_equal(Tensor *A, Tensor *B, float v);
void gpu_less_equal(Tensor *A, Tensor *B, Tensor *C);
void gpu_equal(Tensor *A, Tensor *B, float v);
void gpu_equal(Tensor *A, Tensor *B, Tensor *C);
void gpu_not_equal(Tensor *A, Tensor *B, float v);
void gpu_not_equal(Tensor *A, Tensor *B, Tensor *C);


// Legacy  **************************************************************************************

void gpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,bool mask_zeros=false); // TODO: Legacy
void gpu_deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end,int inc,bool mask_zeros=false); // TODO: Legacy

void gpu_copy_to_gpu(float *nptr,Tensor *B);
void gpu_copy_from_gpu(Tensor *A,float *nptr);
void gpu_copy_gpu(Tensor *A,Tensor *B);

void cpu2gpu(float *dst, const float *src, unsigned long int size, int gpu_device);
void gpu2cpu(float *dst, const float *src, unsigned long int size, int gpu_device);

float* get_gpu_fmem(unsigned long int size, int gpu_device);
void free_gpu_ptr(float *ptr, int gpu_device);
void gpu_mask(Tensor *A,float v);


void gpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);
float* gpu_get_uniforms(int N);  // TODO: What is this?
void gpu_total_sum(Tensor *A, float *tot);

// GPU: Temp
int* get_block_dim(int N, int blockSize);
void copy_cpu2gpu(float* cpu_addresses, float* gpu_addresses, int size, bool delete_cpu);
void gpu_initialize_rd(ReduceDescriptor2 *rd, Tensor *A, Tensor *B, bool reverse=false);

#endif //EDDL_GPU_HW_H
