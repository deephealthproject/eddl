//
// Created by Salva Carrión on 30/09/2019.
//

#ifndef EDDL_GPU_HW_H
#define EDDL_GPU_HW_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(A) int r,c;r=(A->size/MAX_TPB);if (r==0) {r=1;c=A->size;}else {if (A->size%MAX_TPB) r++;c=MAX_TPB;}dim3 dimGrid(r);dim3 dimBlock(c);

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];

// GPU: Temp

// GPU: Comparison
int gpu_equal(Tensor *A, Tensor *B);

// GPU: Core (static)
void gpu_set(Tensor *A, float v);
void gpu_mask(Tensor *A,float v);

void gpu_copy_to_gpu(float *nptr,Tensor *B);
void gpu_copy_from_gpu(Tensor *A,float *nptr);
void gpu_copy_gpu(Tensor *A,Tensor *B);

void gpu_transpose(Tensor *A, Tensor *B);
void gpu_copy(Tensor *A, Tensor *B);
void gpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc);
void gpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end);

// GPU: Generator
void gpu_rand_uniform(Tensor *A, float v);
void gpu_rand_signed_uniform(Tensor *A, float v);
void gpu_rand_binary(Tensor *A, float v);
void gpu_rand_normal(Tensor *A, float m, float s);

// GPU: Math (in-place)
void gpu_abs(Tensor *A);
void gpu_acos(Tensor *A);
void gpu_add(Tensor *A, float v);
void gpu_asin(Tensor *A);
void gpu_atan(Tensor *A);
void gpu_ceil(Tensor *A);
void gpu_clamp(Tensor *A, float min, float max);
void gpu_cos(Tensor *A);
void gpu_cosh(Tensor *A);
void gpu_exp(Tensor *A);
void gpu_floor(Tensor *A);
void gpu_log(Tensor *A);
void gpu_log2(Tensor *A);
void gpu_log10(Tensor *A);
void gpu_logn(Tensor *A, float n);
void gpu_mod(Tensor *A, float v);
void gpu_mult(Tensor *A, float v);
void gpu_normalize(Tensor *A, float min, float max);
void gpu_pow(Tensor *A, float exp);
void gpu_reciprocal(Tensor *A);
void gpu_remainder(Tensor *A, float v);
void gpu_round(Tensor *A);
void gpu_rsqrt(Tensor *A);
void gpu_sigmoid(Tensor *A);
void gpu_sign(Tensor *A);
void gpu_sin(Tensor *A);
void gpu_sinh(Tensor *A);
void gpu_sqr(Tensor *A);
void gpu_sqrt(Tensor *A);
void gpu_tan(Tensor *A);
void gpu_tanh(Tensor *A);
void gpu_trunc(Tensor *A);

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
void gpu_total_sum(Tensor *A,float *tot);
float gpu_sum(Tensor *A);
float gpu_sum_abs(Tensor *A);

// GPU: Reduction
void gpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB);
void gpu_reduceTosum(Tensor *A, Tensor *B, int axis);
//void gpu_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void gpu_delta_reduce(Tensor *A, Tensor *B, vector<int> axis, string mode, bool keepdims,Tensor *C,int incB);
//void gpu_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op,Tensor *C,int incC);
//void gpu_delta_reduced_op(Tensor *A, Tensor *B, vector<int> axis, string op, Tensor *C,int incC);


#endif //EDDL_GPU_HW_H
