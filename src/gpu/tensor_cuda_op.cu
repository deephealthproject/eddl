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

#include <string.h>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../tensor.h"
#include "tensor_cuda.h"
#include "tensor_kernels.h"

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];

using namespace std;

void check_cuda(cudaError_t error, string func)
{
  if ( error!= cudaSuccess)
  {
     fprintf(stderr,"Error in cuda execution in %s\n",func);
     exit(1);
  }
}
void check_cublas(cublasStatus_t status, string func)
{
  if ( status!=  CUBLAS_STATUS_SUCCESS)
  {
     fprintf(stderr,"Error in cublas execution in %s\n",func);
     exit(1);
  }
}

void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC)
{
int device=A->device-DEV_GPU;

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
	k=B->sizes[0];
  ldC=B->sizes[1];
}

check_cublas(cublasSgemm(hcublas[device],trB,trA,m,n,k,&alfa,B->gptr,ldB,A->gptr,ldA,&beta,C->gptr,ldC),"mult2D");

}

void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C)
{

  dim3 dimGrid(A->sizes[0]);
  dim3 dimBlock(A->sizes[1]);

  sum_mat_row<<<dimBlock,dimGrid>>>(A->gptr,B->gptr,C->gptr,A->sizes[0],A->sizes[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}
