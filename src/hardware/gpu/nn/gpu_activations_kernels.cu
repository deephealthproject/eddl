/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Juan Maroñas: jmaronas@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_nn_kernels.h"
#include "../gpu_kernels.h"


__global__ void relu(float *a,float *b,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (a[thread_id_x]>0.0) b[thread_id_x]=a[thread_id_x];
    else b[thread_id_x]=0.0;
   }
}

__global__ void d_relu(float *d,float *i,float *pd,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (i[thread_id_x]>0.0) pd[thread_id_x]=d[thread_id_x];
    else pd[thread_id_x]=0.0;
   }

}

__global__ void softmax(float* E,float* N,float* auxE ,long int sample_ndim, long int n_vals)
{
    float C_value=0;
    long int thread_id_x = threadIdx.x + blockIdx.x*blockDim.x;
    float maxCoef = E[thread_id_x*sample_ndim];
    float actualCoef = 0;
    if (thread_id_x<n_vals)
    {

	    for (long int cA = 1; cA < sample_ndim; cA++)
    		if (E[thread_id_x*sample_ndim+cA] > maxCoef)
    			 maxCoef=E[thread_id_x*sample_ndim+cA];

	    for (long int cA = 0; cA < sample_ndim; cA++)
  		{
  			actualCoef=expf(E[thread_id_x*sample_ndim+cA]-maxCoef);
  			auxE[thread_id_x*sample_ndim+cA]=actualCoef;
  			C_value+=actualCoef;
  		}

      for (long int cA=0; cA < sample_ndim; cA++)
	       N[thread_id_x*sample_ndim+cA]=auxE[thread_id_x*sample_ndim+cA]/C_value;
    }

}



