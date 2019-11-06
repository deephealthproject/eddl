/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es), (jmaronasm@gmail.com)
* All rights reserved
*/


#ifndef EDDL_GPU_HW_H
#define EDDL_GPU_HW_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

#define PRECISION_FLOAT -std::numeric_limits<float>::max()

// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(A) int r,c;r=(A->size/MAX_TPB);if (r==0) {r=1;c=A->size;}else {if (A->size%MAX_TPB) r++;c=MAX_TPB;}dim3 dimGrid(r);dim3 dimBlock(c);

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];

// GPU: Temp

// GPU: Comparison
int gpu_equal(Tensor *A, Tensor *B);

// GPU: Core
void gpu_fill_(Tensor *A, float v);
void gpu_mask(Tensor *A,float v);

void gpu_copy_to_gpu(float *nptr,Tensor *B);
void gpu_copy_from_gpu(Tensor *A,float *nptr);
void gpu_copy_gpu(Tensor *A,Tensor *B);

void gpu_transpose(Tensor *A, Tensor *B);
void gpu_copy(Tensor *A, Tensor *B);
void gpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
void gpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);


// GPU: Create (static)
void gpu_range(Tensor *A, float start, float step);
void gpu_eye(Tensor *A);

// GPU: Transformations
void gpu_shift(Tensor *A, Tensor *B, vector<int> t_shift, int mode, float constant);
void gpu_rotate(Tensor *A, Tensor *B,float angle, vector<int> axis, int mode, float constant);
void gpu_scale(Tensor *A, Tensor *B, int mode, float constant);
void gpu_flip(Tensor *A, Tensor *B, int axis);
void gpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant);
void gpu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant);
void gpu_cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant);

void gpu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant);
void gpu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> axis, int mode, float constant);
void gpu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant);
void gpu_flip_random(Tensor *A, Tensor *B, int axis);
void gpu_crop_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);
void gpu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);
void gpu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant);

// GPU: Generator
void gpu_rand_uniform(Tensor *A, float v);
void gpu_rand_signed_uniform(Tensor *A, float v);
void gpu_rand_binary(Tensor *A, float v);
void gpu_rand_normal(Tensor *A, float m, float s);

// GPU: Math (in-place)
void gpu_inv_(Tensor *A);
void gpu_abs_(Tensor *A);
void gpu_acos_(Tensor *A);
void gpu_add_(Tensor *A, float v);
void gpu_asin_(Tensor *A);
void gpu_atan_(Tensor *A);
void gpu_ceil_(Tensor *A);
void gpu_clamp_(Tensor *A, float min, float max);
void gpu_cos_(Tensor *A);
void gpu_cosh_(Tensor *A);
void gpu_exp_(Tensor *A);
void gpu_floor_(Tensor *A);
void gpu_log_(Tensor *A);
void gpu_log2_(Tensor *A);
void gpu_log10_(Tensor *A);
void gpu_logn_(Tensor *A, float n);
void gpu_mod_(Tensor *A, float v);
void gpu_mult_(Tensor *A, float v);
void gpu_normalize_(Tensor *A, float min, float max);
void gpu_pow_(Tensor *A, float exp);
void gpu_powb_(Tensor *A, float base);
void gpu_reciprocal_(Tensor *A);
void gpu_remainder_(Tensor *A, float v);
void gpu_round_(Tensor *A);
void gpu_rsqrt_(Tensor *A);
void gpu_sigmoid_(Tensor *A);
void gpu_sign_(Tensor *A);
void gpu_sin_(Tensor *A);
void gpu_sinh_(Tensor *A);
void gpu_sqr_(Tensor *A);
void gpu_sqrt_(Tensor *A);
void gpu_tan_(Tensor *A);
void gpu_tanh_(Tensor *A);
void gpu_trunc_(Tensor *A);

// GPU: Math (static)
void gpu_addc(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC);
void gpu_inc(Tensor *A, Tensor *B);
void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC);
void gpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC);
void gpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC);
void gpu_sign2(Tensor *A, Tensor *B); // TODO: Remove
void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);


// GPU: Should be reductions
float gpu_max(Tensor *A);
float gpu_min(Tensor *A);
float gpu_sum(Tensor *A);
void gpu_total_sum(Tensor *A, float *tot);
float gpu_sum_abs(Tensor *A);

// GPU: Reduction
void gpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
void gpu_reduceTosum(Tensor *A, Tensor *B, int axis);
void gpu_reduction(ReduceDescriptor *RD);
void gpu_reduction_back(ReduceDescriptor *RD);
//void gpu_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void gpu_delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void gpu_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
//void gpu_delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);


#endif //EDDL_GPU_HW_H
