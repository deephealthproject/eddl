/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "eddl/hardware/gpu/nn/gpu_nn.h"
#include "eddl/hardware/gpu/nn/gpu_nn_kernels.h"

#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"

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

void gpu_sigmoid(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sigmoid<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_sigmoid");
}

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


void gpu_exp(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  exp<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_exp");
}

void gpu_d_exp(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_exp<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_exp");
}

void gpu_tanh(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  tanh<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_tanh");
}

void gpu_d_tanh(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_tanh<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_tanh");
}


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
