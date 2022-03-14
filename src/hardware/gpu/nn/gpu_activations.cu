/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
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
#include "eddl/descriptors/descriptors.h"



void gpu_relu(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  relu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_relu");
}


void gpu_thresholded_relu(Tensor *A,Tensor *B,float param){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  thresholded_relu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,param,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_thresholded_relu");
}


void gpu_d_thresholded_relu(Tensor *D,Tensor *I,Tensor *PD,float param) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_thresholded_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,param,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_thresholded_relu");
}


void gpu_leaky_relu(Tensor *A,Tensor *B,float param){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  leaky_relu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,param,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_leaky_relu");
}


void gpu_d_leaky_relu(Tensor *D,Tensor *I,Tensor *PD,float param) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_leaky_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,param,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_leaky_relu");
}

void gpu_elu(Tensor *A,Tensor *B,float param){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  elu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,param,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_elu");
}


void gpu_d_elu(Tensor *D,Tensor *I,Tensor *PD,float param) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_elu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,param,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_elu");
}


void gpu_softplus(Tensor *A,Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    softplus<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
    check_cuda(cudaDeviceSynchronize(),"gpu_softplus");
}

void gpu_d_softplus(Tensor *D,Tensor *I,Tensor *PD){
    int device=D->gpu_device;
    cudaSetDevice(device);

    setDims(D)

    d_softplus<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
    check_cuda(cudaDeviceSynchronize(),"gpu_d_softplus");
}

void gpu_softsign(Tensor *A,Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A)

    softsign<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
    check_cuda(cudaDeviceSynchronize(),"gpu_softsign");
}

void gpu_d_softsign(Tensor *D,Tensor *I,Tensor *PD){
    int device=D->gpu_device;
    cudaSetDevice(device);

    setDims(D)

    d_softsign<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
    check_cuda(cudaDeviceSynchronize(),"gpu_d_softsign");
}


void gpu_linear(Tensor *A,Tensor *B,float param){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  linear<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,param,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_linear");
}


void gpu_d_linear(Tensor *D,Tensor *I,Tensor *PD,float param) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_linear<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,param,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_linear");
}

//void gpu_sigmoid(Tensor *A,Tensor *B){
//  int device=A->gpu_device;
//  cudaSetDevice(device);
//
//  setDims(A);
//
//  sigmoid<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
//  check_cuda(cudaDeviceSynchronize(),"gpu_sigmoid");
//}

void gpu_d_sigmoid(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_sigmoid<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_sigmoid");
}

void gpu_hard_sigmoid(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  hard_sigmoid<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_hard_sigmoid");
}

void gpu_d_hard_sigmoid(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_hard_sigmoid<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_hard_sigmoid");
}


//void gpu_exp(Tensor *A,Tensor *B){
//  int device=A->gpu_device;
//  cudaSetDevice(device);
//
//  setDims(A);
//
//  exp<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
//  check_cuda(cudaDeviceSynchronize(),"gpu_exp");
//}

void gpu_d_exp(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_exp<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_exp");
}

//void gpu_tanh(Tensor *A,Tensor *B){
//  int device=A->gpu_device;
//  cudaSetDevice(device);
//
//  setDims(A);
//
//  tanh<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
//  check_cuda(cudaDeviceSynchronize(),"gpu_tanh");
//}

void gpu_d_tanh(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_tanh<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_tanh");
}

// OLD SOFTMAX => DEPRECATED
void gpu_softmax(Tensor *A,Tensor *B){

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;
  r=A->shape[0];
  c=A->shape[1];

  dim3 dimGrid(1);
  dim3 dimBlock(MAX_TPB);

  int i;
  for(i=0;i<r/MAX_TPB;i++) {
    float *aptr=A->ptr+(i*MAX_TPB*c);
    float *bptr=B->ptr+(i*MAX_TPB*c);
    int size=MAX_TPB*c;

    float* aux=gpu_create_tensor(device,size);
    softmax<<<dimGrid,dimBlock>>>(aptr,bptr,aux,c,size);
    check_cuda(cudaDeviceSynchronize(),"gpu_softmax");
    gpu_delete_tensor(device,aux);
  }

  if (r%MAX_TPB) {
    dim3 dimGridm(1);
    dim3 dimBlockm(r%MAX_TPB);
    float *aptr=A->ptr+(i*MAX_TPB*c);
    float *bptr=B->ptr+(i*MAX_TPB*c);
    int size=(r%MAX_TPB)*c;

    float* aux=gpu_create_tensor(device,size);
    softmax<<<dimGridm,dimBlockm>>>(aptr,bptr,aux,c,size);
    check_cuda(cudaDeviceSynchronize(),"gpu_softmax");
    gpu_delete_tensor(device,aux);
  }
}

void gpu_full_softmax(Tensor *A, Tensor *B, int axis, bool stable){
    gpu_full_softmax_nd(A, B, axis, stable);

//    if(axis==1 && A->ndim==2){  // TODO: Temp. This should be generic for n-dimensions
//        gpu_full_softmax_batched(A, B, stable);
//    }else{
//        gpu_full_softmax_nd(A, B, stable, axis);
//    }
}

void gpu_full_softmax_batched(Tensor *A, Tensor *B, bool stable){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Get Bacthes and Softmax dimension (classes)
    int n_batches = A->shape[0];
    int n_features = A->shape[1];

    // Calculate cuda blocks
    int blockSize = MAX_TPB;
    int numBlocks = (n_batches + blockSize - 1) / blockSize; // Same as: ceil(N/threads_block)

    // Calculate derivative of Softmax
    full_softmax_batched<<<numBlocks, blockSize>>>(A->ptr, B->ptr, stable, n_batches, n_features);
    check_cuda(cudaDeviceSynchronize(),"gpu_full_softmax_batched");
}

void gpu_full_softmax_nd(Tensor *A, Tensor *B, int axis, bool stable){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Get values
    int chuck_size = A->shape[axis];
    int n_samples = A->size/chuck_size;
    int inner_stride = A->stride[axis];
    int sample_stride = chuck_size*A->stride[axis];
    int k_stride = (chuck_size-1)*A->stride[axis];

    // Calculate cuda blocks
    int blockSize = MAX_TPB;
    int numBlocks = (n_samples + blockSize - 1) / blockSize; // Same as: ceil(N/threads_block)

    // Calculate derivative of Softmax
    full_softmax_nd<<<numBlocks, blockSize>>>(A->ptr, B->ptr, stable, n_samples, inner_stride, sample_stride, k_stride);
    check_cuda(cudaDeviceSynchronize(),"gpu_full_softmax_nd");
}

void gpu_d_full_softmax(Tensor *D, Tensor *I, Tensor *PD, int axis){
    gpu_d_full_softmax_nd(D, I, PD, axis);

//    if(axis==1 && D->ndim==2){  // TODO: Temp. This should be generic for n-dimensions
//        gpu_d_full_softmax_batched(D, I, PD);
//    }else{
//        gpu_d_full_softmax_nd(D, I, PD, axis);
//    }
}

void gpu_d_full_softmax_batched(Tensor *D, Tensor *I, Tensor *PD){
    int device=D->gpu_device;
    cudaSetDevice(device);

    // Get Bacthes and Softmax dimension (classes)
    int n_batches = D->shape[0];
    int n_features = D->shape[1];

    // Calculate cuda blocks
    int blockSize = MAX_TPB;
    int numBlocks = (n_batches + blockSize - 1) / blockSize; // Same as: ceil(N/threads_block)

    // Calculate derivative of Softmax
    full_d_softmax_batched<<<numBlocks, blockSize>>>(D->ptr, I->ptr, PD->ptr, n_batches, n_features);
    check_cuda(cudaDeviceSynchronize(),"gpu_d_full_softmax_batched");
}

void gpu_d_full_softmax_nd(Tensor *D, Tensor *I, Tensor *PD, int axis){
    int device=D->gpu_device;
    cudaSetDevice(device);

    // Get values
    int chuck_size = D->shape[axis];
    int n_samples = D->size/chuck_size;
    int inner_stride = D->stride[axis];
    int sample_stride = chuck_size*D->stride[axis];
    int k_stride = (chuck_size-1)*D->stride[axis];

    // Calculate cuda blocks
    int blockSize = MAX_TPB;
    int numBlocks = (n_samples + blockSize - 1) / blockSize; // Same as: ceil(N/threads_block)

    // Calculate derivative of Softmax
    full_d_softmax_nd<<<numBlocks, blockSize>>>(D->ptr, I->ptr, PD->ptr, n_samples, inner_stride, sample_stride, k_stride);
    check_cuda(cudaDeviceSynchronize(),"gpu_d_full_softmax_nd");
}