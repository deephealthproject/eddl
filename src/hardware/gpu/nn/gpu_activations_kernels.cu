/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/gpu_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"


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


__global__ void thresholded_relu(float *a,float *b, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (a[thread_id_x]>param) b[thread_id_x]=a[thread_id_x];
    else b[thread_id_x]=0.0;
   }
}

__global__ void d_thresholded_relu(float *d,float *i,float *pd, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (i[thread_id_x]>param) pd[thread_id_x]=d[thread_id_x];
    else pd[thread_id_x]=0.0;
   }

}


__global__ void leaky_relu(float *a,float *b, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (a[thread_id_x]>0.0) b[thread_id_x]=a[thread_id_x];
    else b[thread_id_x]=param*a[thread_id_x];
   }
}

__global__ void d_leaky_relu(float *d,float *i,float *pd, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (i[thread_id_x]>0.0) pd[thread_id_x]=d[thread_id_x];
    else pd[thread_id_x]=param*d[thread_id_x];
   }

}

__global__ void elu(float *a,float *b, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (a[thread_id_x]>0.0) b[thread_id_x]=a[thread_id_x];
    else b[thread_id_x]=param*(expf(a[thread_id_x]) - 1);
   }
}

__global__ void d_elu(float *d,float *i,float *pd, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (i[thread_id_x]>0.0) pd[thread_id_x]=d[thread_id_x];
    else pd[thread_id_x]=(param*expf(i[thread_id_x])) * d[thread_id_x];
   }

}

__global__ void softplus(float *a,float *b,long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        b[thread_id_x] = logf(1 + expf(a[thread_id_x]));
    }
}

__global__ void d_softplus(float *d,float *i,float *pd,long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        pd[thread_id_x] = d[thread_id_x] * 1/(1 + expf(-i[thread_id_x]));
    }
}

__global__ void softsign(float *a,float *b,long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        b[thread_id_x] = a[thread_id_x] / (1 + abs(a[thread_id_x]));
    }
}

__global__ void d_softsign(float *d,float *i,float *pd,long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        float denom = 1 + abs(i[thread_id_x]);
        pd[thread_id_x] = d[thread_id_x] * (1/(denom*denom));
    }
}

__global__ void linear(float *a,float *b, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    b[thread_id_x] = param * a[thread_id_x];
   }
}

__global__ void d_linear(float *d,float *i,float *pd, float param, long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    pd[thread_id_x] = param * d[thread_id_x];
   }

}
__global__ void sigmoid(float *a,float *b,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    b[thread_id_x]=1/(1+expf(-a[thread_id_x]));
  }
}

__global__ void d_sigmoid(float *d,float *i,float *pd,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    pd[thread_id_x]+=d[thread_id_x]*((1-i[thread_id_x])*i[thread_id_x]);
   }

}

__global__ void hard_sigmoid(float *a,float *b,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (a[thread_id_x] > 2.5) b[thread_id_x] = 1.0;
    else if (a[thread_id_x] < -2.5) b[thread_id_x] = 0.0;
    else b[thread_id_x] = (a[thread_id_x] * 0.2) + 0.5;
  }
}

__global__ void d_hard_sigmoid(float *d,float *i,float *pd,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    if (i[thread_id_x] < -2.5 || i[thread_id_x] > 2.5) pd[thread_id_x] = 0.0;
    else pd[thread_id_x] = 0.2 * d[thread_id_x];
   }
}

__global__ void exp(float *a,float *b,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    b[thread_id_x] = expf(a[thread_id_x]);
  }
}

__global__ void d_exp(float *d,float *i,float *pd,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    pd[thread_id_x] = d[thread_id_x] * i[thread_id_x];
   }
}

__global__ void tanh(float *a,float *b,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    float p=expf(a[thread_id_x]);
    float n=expf(-a[thread_id_x]);
    b[thread_id_x]=(p-n)/(p+n);
  }
}

__global__ void d_tanh(float *d,float *i,float *pd,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size){
    pd[thread_id_x]+=d[thread_id_x]*(1-(i[thread_id_x]*i[thread_id_x]));
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
