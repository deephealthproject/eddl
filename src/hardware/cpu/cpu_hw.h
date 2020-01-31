/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_CPU_HW_H
#define EDDL_CPU_HW_H

#include "../../tensor/tensor.h"
#include "../../tensor/tensor_reduction.h"
#include "../../descriptors/descriptors.h"

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

// CPU: Core (static)
void cpu_transpose(Tensor *A, Tensor *B);
void cpu_copy(Tensor *A, Tensor *B);

void cpu_fill_(Tensor *A, float v);
void cpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);

void cpu_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void cpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
void cpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);
void cpu_deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);

void cpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative);

void cpu_repeat(Tensor *A, Tensor *B, vector<int> size);
void cpu_d_repeat(Tensor *D, Tensor *A, vector<int> size);

// CPU: Create
void cpu_range(Tensor *A, float min, float step);
void cpu_eye(Tensor *A, int offset);

// CPU: Generator
void cpu_rand_uniform(Tensor *A, float v);
void cpu_rand_signed_uniform(Tensor *A, float v);
void cpu_rand_binary(Tensor *A, float v);
void cpu_rand_normal(Tensor *A, float m, float s, bool fast_math);  // TODO: Don't like it

// CPU: Data transformations (2D Optimized) ********************************************
void cpu_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant);
void cpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant);
void cpu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant);
void cpu_flip(Tensor *A, Tensor *B, int axis);
void cpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse);
void cpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant);

// CPU: Data augmentations (2D Optimized) ********************************************
void cpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant);
void cpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant);
void cpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void cpu_flip_random(Tensor *A, Tensor *B, int axis);
void cpu_crop_random(Tensor *A, Tensor *B);
void cpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void cpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);

// CPU: Math (in-place)
void cpu_abs_(Tensor *A);
void cpu_acos_(Tensor *A);
void cpu_add_(Tensor *A, float v);
void cpu_asin_(Tensor *A);
void cpu_atan_(Tensor *A);
void cpu_ceil_(Tensor *A);
void cpu_clamp_(Tensor *A, float min, float max);
void cpu_cos_(Tensor *A);
void cpu_cosh_(Tensor *A);
void cpu_exp_(Tensor *A);
void cpu_inv_(Tensor *A, float v);
void cpu_floor_(Tensor *A);
void cpu_log_(Tensor *A);
void cpu_log2_(Tensor *A);
void cpu_log10_(Tensor *A);
void cpu_logn_(Tensor *A, float n);
void cpu_mod_(Tensor *A, float v);
void cpu_mult_(Tensor *A, float v);
void cpu_normalize_(Tensor *A, float min, float max);
void cpu_pow_(Tensor *A, float exp);
void cpu_powb_(Tensor *A, float base);
void cpu_reciprocal_(Tensor *A);
void cpu_remainder_(Tensor *A, float v);
void cpu_round_(Tensor *A);
void cpu_rsqrt_(Tensor *A);
void cpu_sigmoid_(Tensor *A);
void cpu_sign_(Tensor *A);
void cpu_sin_(Tensor *A);
void cpu_sinh_(Tensor *A);
void cpu_sqr_(Tensor *A);
void cpu_sqrt_(Tensor *A);
void cpu_tan_(Tensor *A);
void cpu_tanh_(Tensor *A);
void cpu_trunc_(Tensor *A);

// CPU: Math (static)
void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
void cpu_inc(Tensor *A, Tensor *B);
void cpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
void cpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
void cpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
void cpu_sign2(Tensor *A, Tensor *B); // TODO: Remove
void cpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void cpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

// CPU: Should be reductions
float cpu_max(Tensor *A);
float cpu_min(Tensor *A);
float cpu_sum(Tensor *A);
float cpu_sum_abs(Tensor *A);

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
bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan);
void cpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan);
void cpu_greater(Tensor *A, Tensor *B, Tensor *C);
void cpu_greater_equal(Tensor *A, Tensor *B, Tensor *C);
void cpu_less(Tensor *A, Tensor *B, Tensor *C);
void cpu_less_equal(Tensor *A, Tensor *B, Tensor *C);
void cpu_equal(Tensor *A, Tensor *B, Tensor *C);
void cpu_not_equal(Tensor *A, Tensor *B, Tensor *C);

// Legacy
int cpu_equal2(Tensor *A, Tensor *B, float epsilon);



#endif //EDDL_CPU_HW_H
