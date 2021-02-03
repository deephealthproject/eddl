/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"

__global__ void accuracy(float* T, float* N,float* acc,long int cols, long int total_ops, int* MC_err){

long int thread_id_x = threadIdx.x + blockIdx.x*blockDim.x;
long int result_t=T[thread_id_x*cols];
float result_n=N[thread_id_x*cols];

long int row_max_t=0;
long int row_max_n=0;

long int aux_t;
float aux_n;
if (thread_id_x < total_ops)
{
  for(long int i=1;i<cols;i++)
  {
   aux_t=T[thread_id_x*cols+i];
   aux_n=N[thread_id_x*cols+i];

	if (aux_t>result_t)
	 {
  		result_t=aux_t;
      row_max_t=i;
   }
  if (aux_n>result_n)
	 {
		result_n=aux_n;
    row_max_n=i;
   }
  }

  acc[thread_id_x]=row_max_t;
  atomicAdd(MC_err,(long int)(row_max_t==row_max_n));
}

}

__global__ void bin_accuracy(float* T, float* N, int size, int* acc){

long int thread_id_x = threadIdx.x + blockIdx.x*blockDim.x;

if (thread_id_x < size)
{
  if ((T[thread_id_x ]==1.0)&&(N[thread_id_x]>0.5))
    atomicAdd(acc,1);
  else if ((T[thread_id_x ]==0.0)&&(N[thread_id_x]<=0.5))
    atomicAdd(acc,1);

}

}
