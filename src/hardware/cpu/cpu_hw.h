/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_CPU_HW_H
#define EDDL_CPU_HW_H

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

// CPU: Comparison
int cpu_equal(Tensor *A, Tensor *B, float epsilon);

// CPU: Core (static)
void cpu_transpose(Tensor *A, Tensor *B);
void cpu_copy(Tensor *A, Tensor *B);
void cpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
void cpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);
void cpu_repeat(Tensor *A, Tensor *B, vector<int> size);
void cpu_d_repeat(Tensor *D, Tensor *A, vector<int> size);

// CPU: Create
void cpu_range(Tensor *A, float min, float step);

// CPU: Generator
void cpu_rand_uniform(Tensor *A, float v);
void cpu_rand_signed_uniform(Tensor *A, float v);
void cpu_rand_binary(Tensor *A, float v);
void cpu_rand_normal(Tensor *A, float m, float s, bool fast_math);  // TODO: Don't like it

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
void cpu_inv_(Tensor *A);
void cpu_floor_(Tensor *A);
void cpu_log_(Tensor *A);
void cpu_log2_(Tensor *A);
void cpu_log10_(Tensor *A);
void cpu_logn_(Tensor *A, float n);
void cpu_mod_(Tensor *A, float v);
void cpu_mult_(Tensor *A, float v);
void cpu_normalize_(Tensor *A, float min, float max);
void cpu_pow_(Tensor *A, float exp);
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
void cpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
void cpu_reduceTosum(Tensor *A, Tensor *B, int axis);
void cpu_reduction(ReduceDescriptor *RD);
void cpu_reduction_back(ReduceDescriptor *RD);
//void cpu_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void cpu_delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void cpu_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
//void cpu_delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);


#endif //EDDL_CPU_HW_H
