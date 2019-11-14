/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_nn.h"
#include "gpu_nn_kernels.h"

#include "../gpu_hw.h"
#include "../gpu_tensor.h"
#include "../gpu_kernels.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


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


void gpu_lrelu(Tensor *A,Tensor *B,float param){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  lrelu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,param,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


void gpu_d_lrelu(Tensor *D,Tensor *I,Tensor *PD,float param) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_lrelu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,param,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_d_relu");
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
