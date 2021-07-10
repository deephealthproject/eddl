/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"

#include "eddl/hardware/gpu/gpu_tensor.h"

#include "eddl/tensor/tensor.h"

#define VERBOSE 0


void gpu_permute_channels_last(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);
  bn_permute_channels_last<<<dimGrid,dimBlock>>>(A->ptr, B->ptr,A->shape[0],A->shape[1],A->shape[2],A->shape[3],A->size);
  check_cuda(cudaDeviceSynchronize(),"bn_permute_channels_last");
}

void gpu_permute_channels_first(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);
  bn_permute_channels_first<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,B->shape[0],B->shape[1],B->shape[2],B->shape[3],B->size);
  check_cuda(cudaDeviceSynchronize(),"bn_permute_channels_first");
}

void gpu_permute_batch_last(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);
  bn_permute_batch_last<<<dimGrid,dimBlock>>>(A->ptr, B->ptr,A->shape[0],A->shape[1],A->shape[2],A->shape[3],A->size);
  check_cuda(cudaDeviceSynchronize(),"bn_permute_batch_last");
}

void gpu_permute_batch_first(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);
  bn_permute_batch_first<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,B->shape[0],B->shape[1],B->shape[2],B->shape[3],B->size);
  check_cuda(cudaDeviceSynchronize(),"bn_permute_batch_first");
}

void gpu_batchnorm_forward(int gpu_device, int b, int z, int rc,
        float *input, float *output, float *opa,
        float *global_mean, float *global_variance,
        float *affine_g, float *affine_b,
        float *mean, float *variance,
        bool trmode, float epsilon, float momentum)
{
    cudaSetDevice(gpu_device);
    int rcz = rc * z;
    int num_blocks = rcz / batch_norm_block_size;
    if (rcz % batch_norm_block_size) num_blocks++;
    int num_blocks_z = z / batch_norm_block_size;
    if (z % batch_norm_block_size) num_blocks_z++;
    if (trmode) {
        // compute mean and variance
        // for (int j = 0; j < z; j++) mean[j] = variance[j] = 0.0;
        check_cuda(cudaMemset(mean, 0, z * sizeof(float)), "gpu_batchnorm_forward");
        check_cuda(cudaMemset(variance, 0, z * sizeof(float)), "gpu_batchnorm_forward");
        // compute mean and variance
        gpu_batchnorm_forward_1<<<num_blocks, batch_norm_block_size>>>(b, rc, rcz, input, mean, variance);
        gpu_batchnorm_forward_2<<<num_blocks_z, batch_norm_block_size>>>(z, 1.0 / (b * rc), mean, variance, momentum, global_mean, global_variance, epsilon);
        // normalization
        gpu_batchnorm_forward_3<<<num_blocks, batch_norm_block_size>>>(b, rc, rcz, input, mean, variance, affine_g, affine_b, opa, output);
    } else {
        gpu_batchnorm_forward_2<<<num_blocks_z, batch_norm_block_size>>>(z, 1.0 / (b * rc), NULL, variance, momentum, NULL, global_variance, epsilon);
        // normalization
        gpu_batchnorm_forward_3<<<num_blocks, batch_norm_block_size>>>(b, rc, rcz, input, global_mean, variance, affine_g, affine_b, opa, output);
    }
}

void gpu_batchnorm_backward(int gpu_device, int b, int z, int rc, float *delta, float *opa, float *pdelta, float *gbn_g, float *gbn_b, float *bn_g, float *variance, float *mean1, float *mean2)
{
    cudaSetDevice(gpu_device);
    int rcz = rc * z;
    int num_blocks = rcz / batch_norm_block_size;
    if (rcz % batch_norm_block_size) num_blocks++;
    int num_blocks_z = z / batch_norm_block_size;
    if (z % batch_norm_block_size) num_blocks_z++;
    float N = b * rc;
    // for (int j = 0; j < z; j++) mean1[j] = mean2[j] = 0.0;
    check_cuda(cudaMemset(mean1, 0, z * sizeof(float)), "gpu_batchnorm_backward");
    check_cuda(cudaMemset(mean2, 0, z * sizeof(float)), "gpu_batchnorm_backward");
    if (bn_g != NULL) {
        // compute mean
        gpu_batchnorm_backward_1<<<num_blocks, batch_norm_block_size>>>(b, rc, rcz, delta, opa, bn_g, mean1, mean2);
        gpu_batchnorm_backward_2<<<num_blocks_z, batch_norm_block_size>>>(z, 1.0 / (b * rc), mean1, mean2, gbn_g, gbn_b, bn_g);
    } else {
        // compute mean
        gpu_batchnorm_backward_1<<<num_blocks, batch_norm_block_size>>>(b, rc, rcz, delta, opa, NULL, mean1, mean2);
        gpu_batchnorm_backward_2<<<num_blocks_z, batch_norm_block_size>>>(z, 1.0 / (b * rc), mean1, mean2, NULL, NULL, NULL);
    }
    gpu_batchnorm_backward_3<<<num_blocks, batch_norm_block_size>>>(b, rc, rcz, delta, opa, pdelta, mean1, mean2, variance);
}
