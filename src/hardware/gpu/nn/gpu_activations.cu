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
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


void gpu_sigmoid(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sigmoid<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}

void gpu_d_sigmoid(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_sigmoid<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


void gpu_softmax(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);


/*
dimBlock.x=sp->row;
 dimGrid.x=1;
 int ops = sp->col*sp->row;
int sample_ndim=sp->col;

double alfa=1;
float* auxE=NULL;
  ops=sp->row;
          auxE = makeTensor(sp->col,sp->row);
          set_sc(auxE, 0.0, sp);
  	Softmax<<<dimBlock,dimGrid>>>(E,N,auxE,sample_ndim,ops);
*/

  int r,c;

  r=A->shape[0];
  c=A->shape[1];

  dim3 dimGrid(1);
  dim3 dimBlock(r);

  float* aux=gpu_create_tensor(device,A->size);
  softmax<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,aux,c,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
  gpu_delete_tensor(device,aux);
}


void gpu_d_softmax(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}
