/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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
    if (i[thread_id_x]>0.0) pd[thread_id_x]+= d[thread_id_x];
    else pd[thread_id_x]+= 0.0;
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
    if (i[thread_id_x]>param) pd[thread_id_x]+= d[thread_id_x];
    else pd[thread_id_x]+= 0.0;
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
    if (i[thread_id_x]>0.0) pd[thread_id_x]+= d[thread_id_x];
    else pd[thread_id_x]+= param*d[thread_id_x];
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
    if (i[thread_id_x]>0.0) pd[thread_id_x]+= d[thread_id_x];
    else pd[thread_id_x]+= (param*expf(i[thread_id_x])) * d[thread_id_x];
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
        pd[thread_id_x] += d[thread_id_x] * 1/(1 + expf(-i[thread_id_x]));
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
        pd[thread_id_x]+=  d[thread_id_x] * (1/(denom*denom));
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
    pd[thread_id_x]+=  param * d[thread_id_x];
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
    if (i[thread_id_x] < -2.5 || i[thread_id_x] > 2.5) pd[thread_id_x]+=  0.0;
    else pd[thread_id_x]+=  0.2 * d[thread_id_x];
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
    pd[thread_id_x]+=  d[thread_id_x] * i[thread_id_x];
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


// TODO: DEPRECATED
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

__global__ void full_softmax_batched(float *A, float *B, bool stable, unsigned int n_batches, unsigned int n_features){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Batch index

    if (thread_id_x < n_batches){
        unsigned int batch_i = thread_id_x; // Alias

        // Contiguous data
        unsigned int start = batch_i*n_features;
        unsigned int end = start+n_features;

        // Numerical stability (opt.)
        // stable => first value, no stable => 0.0f
        float max_value = GPU_LOWEST_FLOAT;
        if(stable){
            for(unsigned int j=start; j<end; j++){
                if (A[j] > max_value) { max_value = A[j]; }
            }
        }

        // Numerator
        float denominator = GPU_EPS_FLOAT;
        for(unsigned int j=start; j<end; j++){
            float value = expf(A[j] - max_value);  // Highest number should be zero
            B[j] = value;
            denominator += value;
        }

        // Softmax
        for(unsigned int j=start; j<end; j++){
            B[j] /= denominator;
        }
    }
}

__global__ void full_d_softmax_batched(float *D, float *I, float *PD, unsigned int n_batches, unsigned int n_features){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Batch index

    if (thread_id_x < n_batches){
        unsigned int batch_i = thread_id_x; // Alias
        float* SM = I; // Alias (softmax)

        // Contiguous data
        unsigned int start = batch_i*n_features;

        // 1) Compute Jacobbian matrix: DS=[ NxN ]  // DjSi
        // 2) Compute delta: D * DS = (1,n)x(n,n)=(1,n)
        // 2.1) Dot product: PD[i] = Dj*DjSi = D0*D0Di + D1*D1Di + ... Dn*DnSi
        for(int i=0; i<n_features; i++){  // Rows
            for(int j=0; j<n_features; j++){  // Cols

                // Derivative
                float DjSi = SM[start+i] * (float)(i==j) - SM[start+j]*SM[start+i];
                PD[start+i] += D[start+j] * DjSi;

            }
        }
    }
}



__global__ void full_softmax_nd(float *A, float *B, bool stable, int n_samples, int inner_stride, int sample_stride, int k_stride){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Batch index

    if (thread_id_x < n_samples){
        int si = (int)thread_id_x; // Alias

        int start_b = si % inner_stride + si/inner_stride * sample_stride;
        int end_b = start_b + k_stride;

        // Numerical stability (opt.)
        // stable => first value, no stable => 0.0f
        float max_value = GPU_LOWEST_FLOAT;
        if(stable){
            for (int i = start_b; i <= end_b; i += inner_stride) {
                if (A[i] > max_value) { max_value = A[i]; }
            }
        }

        // Numerator
        float denominator = GPU_EPS_FLOAT;
        for (int i = start_b; i <= end_b; i += inner_stride) {
            float value = expf(A[i] - max_value);  // Highest number should be zero
            B[i] = value;
            denominator += value;
        }

        // Softmax
        for (int i = start_b; i <= end_b; i += inner_stride) {
            B[i] /= denominator;
        }
    }
}

__global__ void full_d_softmax_nd(float *D, float *I, float *PD, int n_samples, int inner_stride, int sample_stride, int k_stride){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Batch index

    if (thread_id_x < n_samples){
        float* SM = I; // Alias (softmax)
        int si = (int)thread_id_x; // Alias

        int start_b = si % inner_stride + si/inner_stride * sample_stride;
        int end_b = start_b + k_stride;

        // 1) Compute Jacobbian matrix: DS=[ NxN ]  // DjSi
        // 2) Compute delta: D * DS = (1,n)x(n,n)=(1,n)
        // 2.1) Dot product: PD[i] = Dj*DjSi = D0*D0Di + D1*D1Di + ... Dn*DnSi
        for (int i = start_b; i <= end_b; i += inner_stride) {  // Rows
            for (int j = start_b; j <= end_b; j += inner_stride) {  // Cols

                // Derivative
                float DjSi = SM[i] * (float)(i==j) - SM[j]*SM[i];
                PD[i] += D[j] * DjSi;

            }
        }
    }
}
