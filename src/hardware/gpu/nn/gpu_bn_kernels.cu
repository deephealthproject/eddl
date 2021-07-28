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

#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"


__global__ void bn_permute_channels_last(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(bo*(r*c*z))+(ro*(c*z))+(co*z)+zo;
    dest[pos]=src[thread_id_x];
  }
}

__global__ void bn_permute_channels_first(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(bo*(r*c*z))+(ro*(c*z))+(co*z)+zo;
    dest[thread_id_x]=src[pos];
  }
}


__global__ void bn_permute_batch_last(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(zo*(r*c*b))+(ro*(c*b))+(co*b)+bo;
    dest[pos]=src[thread_id_x];
  }
}

__global__ void bn_permute_batch_first(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(zo*(r*c*b))+(ro*(c*b))+(co*b)+bo;
    dest[thread_id_x]=src[pos];
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
            variance[j] = sqrt(variance[j] + epsilon); // use current batch variance in training
        } else {
            variance[j] = sqrt(global_variance[j] + epsilon); // use global variance in inference
        }
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
            float o = (input[p] - m) / v;
            if (affine_g != nullptr){
                opa[p] = o;
                output[p] = o * affine_g[j] + affine_b[j]; // apply the affine transformation
            } else {
                output[p] = o;
            }
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
            d = d / variance[j]; // step 7
            // delta[p] = d;
            pdelta[p] += d;
        }
    }
}
