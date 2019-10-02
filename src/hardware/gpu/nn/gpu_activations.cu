#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../tensor_cuda.h"
#include "../tensor_kernels.h"
#include "../gpu_hw.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(A) int r,c;r=(A->size/MAX_TPB);if (r==0) {r=1;c=A->size;}else {if (A->size%MAX_TPB) r++;c=MAX_TPB;}dim3 dimGrid(r);dim3 dimBlock(c);


extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];

void gpu_relu(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

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
