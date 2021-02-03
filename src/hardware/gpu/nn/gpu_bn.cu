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




















/////////
