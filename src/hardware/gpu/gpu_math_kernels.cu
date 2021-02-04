/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/gpu_kernels.h"


 __global__ void gpu_abs(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = fabsf(A[thread_id_x]);
    }
}

 __global__ void gpu_acos(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = acosf(A[thread_id_x]);
    }
}

 __global__ void gpu_add(float *A, float *B, long int size, float v){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = A[thread_id_x] + v;
    }
}

 __global__ void gpu_asin(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = asinf(A[thread_id_x]);
    }
}

 __global__ void gpu_atan(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = atanf(A[thread_id_x]);
    }

}
 __global__ void gpu_ceil(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = ceilf(A[thread_id_x]);
    }
}

 __global__ void gpu_clamp(float *A, float *B, long int size, float min, float max){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        if (A[thread_id_x] < min){
            B[thread_id_x] = min;
        }else if(A[thread_id_x] > max){
            B[thread_id_x] = max;
        }else {
            B[thread_id_x] = A[thread_id_x];
        }
}

 __global__ void gpu_cos(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = cosf(A[thread_id_x]);
    }
}

 __global__ void gpu_cosh(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = coshf(A[thread_id_x]);
    }
}

 __global__ void gpu_exp(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = expf(A[thread_id_x]);
    }
}

 __global__ void gpu_floor(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = floorf(A[thread_id_x]);
    }
}

 __global__ void gpu_log(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = logf(A[thread_id_x]);
    }
}

 __global__ void gpu_log2(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = log2f(A[thread_id_x]);
    }
}

 __global__ void gpu_log10(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = log10f(A[thread_id_x]);
    }
}

 __global__ void gpu_logn(float *A, float *B, long int size, float n){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = logf(A[thread_id_x])/logf(n);
    }
}

 __global__ void gpu_mod(float *A, float *B, long int size, float v){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = fmodf(A[thread_id_x], v);
    }
}

 __global__ void gpu_inv(float *A, float *B, long int size, float v){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = v/A[thread_id_x];
    }
}

 __global__ void gpu_mult(float *A, float *B, long int size, float v){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = A[thread_id_x] * v;
    }
}

 __global__ void gpu_normalize(float *A, float *B, long int size, float min_ori, float max_ori, float min, float max){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = (max-min)/(max_ori-min_ori) * (A[thread_id_x]-min_ori) + min;
    }
}

 __global__ void gpu_pow(float *A, float *B, long int size, float exp){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = powf(A[thread_id_x], exp);
    }
}

 __global__ void gpu_powb(float *A, float *B, long int size, float base){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = powf(base, A[thread_id_x]);
    }
}


 __global__ void gpu_remainder(float *A, float *B, long int size, float v){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = fmod((v + fmod(A[thread_id_x], v)), v);
    }
}

 __global__ void gpu_round(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = roundf(A[thread_id_x]);
    }
}

 __global__ void gpu_rsqrt(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = 1.0f/sqrtf(A[thread_id_x]);
    }
}

 __global__ void gpu_sigmoid(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = 1.0f/(1.0f + ::expf(-A[thread_id_x]));
    }
}

 __global__ void gpu_sign(float *A, float *B, long int size, float zero_sign){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        if(A[thread_id_x] > 0.0f){
            B[thread_id_x] = 1.0f;
        }else if(A[thread_id_x] < 0.0f){
            B[thread_id_x] = -1.0f;
        }else{
            B[thread_id_x] = zero_sign;
        }
    }
}

 __global__ void gpu_sin(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = sinf(A[thread_id_x]);
    }
}

 __global__ void gpu_sinh(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = sinhf(A[thread_id_x]);
    }
}

 __global__ void gpu_sqr(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = A[thread_id_x] * A[thread_id_x];
    }
}

 __global__ void gpu_sqrt(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = sqrtf(A[thread_id_x]);
    }
}

 __global__ void gpu_tan(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = tanf(A[thread_id_x]);
    }
}

 __global__ void gpu_tanh(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = tanhf(A[thread_id_x]);
    }
}

 __global__ void gpu_trunc(float *A, float *B, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = truncf(A[thread_id_x]);
    }
}


// CPU: Math (static) ***************************

 __global__ void gpu_add(float scA, float *A, float scB, float *B, float *C, long int incC, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size) {
        if (incC) C[thread_id_x] += scA * A[thread_id_x] + scB * B[thread_id_x];
        else C[thread_id_x] = scA * A[thread_id_x] + scB * B[thread_id_x];
    }
}

 __global__ void gpu_el_mult(float *A, float *B, float *C, long int incC, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        if (incC) C[thread_id_x] += A[thread_id_x] * B[thread_id_x];
        else C[thread_id_x] = A[thread_id_x] * B[thread_id_x];
    }
}

 __global__ void gpu_el_div(float *A, float *B, float *C, long int incC, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        if (incC) C[thread_id_x] += A[thread_id_x]/(B[thread_id_x]);
        else C[thread_id_x] = A[thread_id_x]/(B[thread_id_x]);
    }
}


 __global__ void gpu_sum2D_rowwise(float *A, float *B, float *C, long int rows,long int cols){
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops){
        C[thread_id_x]=A[thread_id_x]+B[thread_id_x%cols];
    }
}

 __global__ void gpu_sum2D_colwise(float *A, float *B, float *C, long int rows,long int cols){
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops){
        C[thread_id_x]=A[thread_id_x]+B[thread_id_x/cols];
    }
}


 __global__ void gpu_reduce_sum2D(float *A,float *B,long int rows,long int cols,long int axis){
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+(blockDim.x*blockIdx.x);

    if (thread_id_x < ops){
        if (axis==0)
            atomicAdd(&(B[thread_id_x%cols]),A[thread_id_x]);
        else
            atomicAdd(&(B[thread_id_x/cols]),A[thread_id_x]);
    }

}

__global__ void gpu_maximum(float* A, float* B, float v, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = max(A[thread_id_x], v);
    }
 }

__global__ void gpu_maximum(float* A, float* B, float* C, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        C[thread_id_x] = max(A[thread_id_x], B[thread_id_x]);
    }
 }

__global__ void gpu_minimum(float* A, float* B, float v, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        B[thread_id_x] = min(A[thread_id_x], v);
    }
 }

__global__ void gpu_minimum(float* A, float* B, float* C, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        C[thread_id_x] = min(A[thread_id_x], B[thread_id_x]);
    }
 }

// new batchnorm implementation

__global__ void gpu_batchnorm_forward_1(int b, int rc, int rcz, float *input, float *mean, float *variance)
{
    // for (int k = 0; k < rcz; k += batch_norm_block_size)
    int k = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (k < rcz) {
        int j = k / rc;
        float m = 0, v = 0;
        for (int i = 0, p = k; i < b; i++, p += rcz) {
            // for (int l = 0; l < batch_norm_block_size && k + l < rcz; l++, p++) {
            float x = input[p];
            m += x;
            v += x * x;
        }
        atomicAdd(mean + j, m);
        atomicAdd(variance + j, v);
    }
}

__global__ void gpu_batchnorm_forward_2(int z, float inv_N, float *mean, float *variance, float momentum, float *global_mean, float *global_variance, float epsilon)
{
    // for (int j = 0; j < z; j++) {
    int j = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (j < z) {
        if (mean != NULL) {
            mean[j] *= inv_N;
            variance[j] = variance[j] * inv_N - mean[j] * mean[j];
            // update global statistics
            if (momentum != 0.0) {
                global_mean[j] = momentum * global_mean[j] + (1.0 - momentum) * mean[j];
                global_variance[j] = momentum * global_variance[j] + (1.0 - momentum) * variance[j];
            }
            variance[j] = 1.0 / sqrt(variance[j] + epsilon);
        } else {
            variance[j] = 1.0 / sqrt(global_variance[j] + epsilon);
        }
    }
}

__global__ void gpu_batchnorm_forward_2b(int z, float *variance, float *global_variance, float epsilon)
{
    // for (int j = 0; j < z; j++) {
    int j = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (j < z) {
        variance[j] = 1.0 / sqrt(global_variance[j] + epsilon);
    }
}

__global__ void gpu_batchnorm_forward_3(int b, int rc, int rcz, float *input, float *mean, float *variance, float *affine_g, float *affine_b, float *opa, float *output)
{
    // for (int k = 0; k < rcz; k += batch_norm_block_size)
    int k = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (k < rcz) {
        int j = k / rc;
        float m = mean[j];
        float v = variance[j];
        for (int i = 0, p = k; i < b; i++, p += rcz) {
            // for (int l = 0; l < batch_norm_block_size && k + l < rcz; l++, p++) {
            float o = (input[p] - m) * v;
            // affine transformation
            if (affine_g != NULL) {
                output[p] = opa[p] * affine_g[j] + affine_b[j];
                opa[p] = input[p];
            } else output[p] = o;
        }
    }
}

__global__ void gpu_batchnorm_backward_1(int b, int rc, int rcz, float *delta, float *opa, float *bn_g, float *mean1, float *mean2)
{
    // for (int k = 0; k < rcz; k += batch_norm_block_size)
    int k = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (k < rcz) {
        int j = k / rc;
        float m1 = 0, m2 = 0;
        for (int i = 0, p = k; i < b; i++, p += rcz) {
            // for (int l = 0; l < batch_norm_block_size && k + l < rcz; l++, p++) {
            m1 += delta[p] * opa[p]; // step 1 & 2
            m2 += delta[p]; // step 4
            if (bn_g != NULL) delta[p] *= bn_g[j]; // affine
        }
        atomicAdd(mean1 + j, m1);
        atomicAdd(mean2 + j, m2);
    }
}

__global__ void gpu_batchnorm_backward_2(int z, float inv_N, float *mean1, float *mean2, float *gbn_g, float *gbn_b, float *bn_g)
{
    // for (int j = 0; j < z; j++) {
    int j = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (j < z) {
        if (bn_g != NULL) { // affine
            float m1 = mean1[j] * inv_N;
            float m2 = mean2[j] * inv_N;
            gbn_g[j] += m1;
            gbn_b[j] += m2;
            mean1[j] = m1 * bn_g[j];
            mean2[j] = m2 * bn_g[j];
        } else {
            mean1[j] *= inv_N;
            mean2[j] *= inv_N;
        }
    }
}

__global__ void gpu_batchnorm_backward_3(int b, int rc, int rcz, float *delta, float *opa, float *pdelta, float *mean1, float *mean2, float *variance)
{
    // for (int k = 0; k < rcz; k += batch_norm_block_size)
    int k = blockIdx.x * batch_norm_block_size + threadIdx.x;
    if (k < rcz) {
        int j = k / rc;
        for (int i = 0, p = k; i < b; i++, p += rcz) {
            // for (int l = 0; l < batch_norm_block_size && k + l < rcz; l++, p++) {
            float o = opa[p] * mean1[j] + mean2[j]; // step 3 & 5
            // opa[p] = o;
            float d = delta[p] - o; // step 6
            // delta[p] = d / variance[j]; // step 7
            pdelta[p] += d / variance[j];
        }
    }
}
