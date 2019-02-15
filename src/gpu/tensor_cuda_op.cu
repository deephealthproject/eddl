// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../tensor.h"
#include "tensor_cuda.h"
#include "tensor_kernels.h"

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];


void check_cublas(cublasStatus_t status, const char *f)
{
  if ( status!=  CUBLAS_STATUS_SUCCESS)
  {
     fprintf(stderr,"Error in cublas execution in %s\n",f);
     exit(1);
  }
}

void check_curand(curandStatus_t status, const char *f)
{
  if ( status!=  CURAND_STATUS_SUCCESS)
  {
     fprintf(stderr,"Error in curand execution in %s\n",f);
     exit(1);
  }
}


///////////////////////////////////////////
void gpu_set(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  set<<<dimGrid,dimBlock>>>(A->gptr,v,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"set");

}

///////////////////////////////////////////
void gpu_mult(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  mult<<<dimGrid,dimBlock>>>(A->gptr,v,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"mult");

}
///////////////////////////////////////////
void gpu_sum(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  sum<<<dimGrid,dimBlock>>>(A->gptr,v,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"sum");

}

///////////////////////////////////////////
void gpu_log(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  log<<<dimGrid,dimBlock>>>(A->gptr,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"log");

}

///////////////////////////////////////////
void gpu_exp(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  exp<<<dimGrid,dimBlock>>>(A->gptr,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"exp");

}

///////////////////////////////////////////
void gpu_sqrt(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  sqrt<<<dimGrid,dimBlock>>>(A->gptr,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"sqrt");

}

///////////////////////////////////////////
void gpu_sqr(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  sqr<<<dimGrid,dimBlock>>>(A->gptr,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"sqr");

}


///////////////////////////////////////////
void gpu_mask(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;

  r=A->sizes[0];
  c=1;

  if (A->dim>1) c=A->tam/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  mask<<<dimGrid,dimBlock>>>(A->gptr,v,A->sizes[0],A->sizes[1]);
  check_cuda(cudaDeviceSynchronize(),"mask");

}

///////////////////////////////////////////
void gpu_copy_to_gpu(float *nptr,Tensor *A)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(A->gptr,nptr,A->tam*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy_to_gpu");
}

///////////////////////////////////////////
void gpu_copy_from_gpu(Tensor *A,float *nptr)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(nptr,A->gptr,A->tam*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy_to_gpu");
}

///////////////////////////////////////////
void gpu_copy_gpu(Tensor *A,Tensor *B)
{
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(B->gptr,A->gptr,A->tam*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
}


///////////////////////////////////////////

void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  float alfa=1.0;
  float beta=(float)incC;

  cublasOperation_t trA = CUBLAS_OP_N;
  cublasOperation_t trB = CUBLAS_OP_N;

  int ldA=A->sizes[1];
  int ldB=B->sizes[1];
  int ldC=B->sizes[1];
  int m=B->sizes[1];
  int n=A->sizes[0];
  int k=B->sizes[0];


  if (tA)
  {
    trA = CUBLAS_OP_T;
  	n=A->sizes[1];
  }
  if (tB)
    {
  	trB = CUBLAS_OP_T;
    m=B->sizes[0];
  	k=B->sizes[1];
    ldC=B->sizes[0];
    }

  check_cublas(cublasSgemm(hcublas[device],trB,trA,m,n,k,&alfa,B->gptr,ldB,A->gptr,ldA,&beta,C->gptr,ldC),"mult2D");

}

///////////////////////////////////////////
void gpu_sum2D(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  int m=A->sizes[1];
  int n=B->sizes[0];
  int ldA=A->sizes[1];
  int ldB=B->sizes[1];
  int ldC=A->sizes[1];

  float alfa=scA;
  float beta=scB;
  float one=1.0;


  if (incC){
    check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,A->gptr,ldA,&one,C->gptr,ldB,C->gptr,ldC),"sum2D");
    check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,B->gptr,ldA,&one,C->gptr,ldB,C->gptr,ldC),"sum2D");
  }
  else
    check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,A->gptr,ldA,&beta,B->gptr,ldB,C->gptr,ldC),"sum2D");

}


///////////////////////////////////////////
void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  dim3 dimGrid(A->sizes[0]);
  dim3 dimBlock(A->sizes[1]);


  sum_mat_row<<<dimGrid,dimBlock>>>(A->gptr,B->gptr,C->gptr,A->sizes[0],A->sizes[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}
///////////////////////////////////////////
void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  dim3 dimGrid(A->sizes[0]);
  dim3 dimBlock(A->sizes[1]);

  sum_mat_col<<<dimGrid,dimBlock>>>(A->gptr,B->gptr,C->gptr,A->sizes[0],A->sizes[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}


///////////////////////////////////////////

void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB)
{

  int device=A->gpu_device;
  cudaSetDevice(device);

  dim3 dimGrid(A->sizes[0]);
  dim3 dimBlock(A->sizes[1]);

  if (!incB) gpu_set(B,0.0);

  reduce_sum2D<<<dimGrid,dimBlock>>>(A->gptr,B->gptr,A->sizes[0],A->sizes[1],axis);


  check_cuda(cudaDeviceSynchronize(),"reduce_sum2D");
}
///////////////////////////////////////////

///////////////////////////////////////////
////// RAND
///////////////////////////////////////////
void gpu_rand_uniform(Tensor *A, float v)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->gptr,A->tam),"gpu_rand_uniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

  gpu_mult(A,v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

}
///////////////////////////////////////////
void gpu_rand_suniform(Tensor *A, float v)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->gptr,A->tam),"gpu_rand_suniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_suniform");

  gpu_mult(A,2*v);
  gpu_sum(A,-v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_suniform");

}

///////////////////////////////////////////
void gpu_rand_gaussian(Tensor *A, float m,float s)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateNormal(random_generator[device],A->gptr,A->tam,m,s),"gpu_rand_gaussian");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_gaussian");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_gaussian");

}

///////////////////////////////////////////
void gpu_rand_binary(Tensor *A, float v)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->gptr,A->tam),"gpu_rand_binary");

  gpu_mask(A,v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_binary");

}













////////////////////////////////////
