/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


int* get_block_dim(int N, int blockSize){
  int* res = new int[2];
  int blocks = (N + blockSize - 1) / blockSize;
  if (N<blockSize) { blockSize = N; }

   res[0] = blocks;
   res[1] = blockSize;
  return res;
}


void gpu_copy_to_gpu(float *nptr,Tensor *A){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(A->ptr,nptr,A->size*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy_to_gpu");
}


void gpu_copy_from_gpu(Tensor *A,float *nptr){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(nptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy_to_gpu");
}


void gpu_copy_gpu(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(B->ptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
}


void gpu_fill(Tensor *A,int aini,int aend,Tensor *B,int bini,int bend,int inc){
  int device=A->gpu_device;
  cudaSetDevice(device);

  int at=A->size/A->shape[0];
  int bt=B->size/B->shape[0];

  int t=1;
  for(int i=2;i<B->ndim;i++)
    t*=B->shape[i];

  int tot=B->shape[0]*(bend-1)*B->shape[1]*t;
  int r,c;

  while (aend-aini>0) {

      if ((aend-aini)>MAX_TPB) r=MAX_TPB;
      else r=(aend-aini);
      c=t;

      dim3 dimGrid(A->shape[0],c);
      dim3 dimBlock(r);

      fill<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,t,aini,at,bini,bt,tot,inc);
      aini+=MAX_TPB;
      bini+=MAX_TPB;

  }

    //check_cuda(cudaDeviceSynchronize(),"fill");

}


void gpu_mask(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  mask<<<dimGrid,dimBlock>>>(A->ptr,v,A->size);
  check_cuda(cudaDeviceSynchronize(),"mask");

}


void gpu_fill_(Tensor *A, float v) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    fill_<<<dimGrid,dimBlock>>>(A->ptr,v,A->size);
    check_cuda(cudaDeviceSynchronize(),"set");
}


void gpu_select(Tensor *A, Tensor *B, const int* indices){
    int device=A->gpu_device;
    cudaSetDevice(device);

    int n;
    // Copy A stride from host to device
    n = A->stride.size();
    int *d_A_stride; cudaMalloc((int**)&d_A_stride, n*sizeof(int));
    cudaMemcpy(d_A_stride, A->stride.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Copy B stride from host to device
    n = B->stride.size();
    int *d_B_stride; cudaMalloc((int**)&d_B_stride, n*sizeof(int));
    cudaMemcpy(d_B_stride, B->stride.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Copy indices from host to device
    n = B->size;
    int *d_indices; cudaMalloc((int**)&d_indices, n*sizeof(int));
    cudaMemcpy(d_indices, indices, n*sizeof(int), cudaMemcpyHostToDevice);

    setDims(B);  // B is the small
    select<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], d_A_stride, d_B_stride, d_indices);
    check_cuda(cudaDeviceSynchronize(), "select");
}

void gpu_select_back(Tensor *A, Tensor *B, const int* indices){
    int device=A->gpu_device;
    cudaSetDevice(device);

    int n;
    // Copy A stride from host to device
    n = A->stride.size();
    int *d_A_stride; cudaMalloc((int**)&d_A_stride, n*sizeof(int));
    cudaMemcpy(d_A_stride, A->stride.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Copy B stride from host to device
    n = B->stride.size();
    int *d_B_stride; cudaMalloc((int**)&d_B_stride, n*sizeof(int));
    cudaMemcpy(d_B_stride, B->stride.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    // Copy indices from host to device
    n = B->size;
    int *d_indices; cudaMalloc((int**)&d_indices, n*sizeof(int));
    cudaMemcpy(d_indices, indices, n*sizeof(int), cudaMemcpyHostToDevice);

    setDims(A);  // A is the small
    select_back<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], d_A_stride, d_B_stride, d_indices);
    check_cuda(cudaDeviceSynchronize(), "select_back");
}