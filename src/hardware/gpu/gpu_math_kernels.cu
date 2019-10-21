/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_kernels.h"


__global__ void abs_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=fabsf(a[thread_id_x]);
}

__global__ void acos_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=acosf(a[thread_id_x]);
}

__global__ void add_(float* a, long int size, float v){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]+=v;
}

__global__ void asin_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=asinf(a[thread_id_x]);
}

__global__ void atan_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=atanf(a[thread_id_x]);
}
__global__ void ceil_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=ceilf(a[thread_id_x]);
}

__global__ void clamp_(float* a, long int size, float min, float max){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        if (a[thread_id_x] < min){
            a[thread_id_x] = min;
        } else if(a[thread_id_x] > max){
            a[thread_id_x] = max;
        }
}

__global__ void cos_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=cosf(a[thread_id_x]);
}

__global__ void cosh_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=coshf(a[thread_id_x]);
}

__global__ void exp_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=expf(a[thread_id_x]);
}

__global__ void floor_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=floorf(a[thread_id_x]);
}

__global__ void log_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=logf(a[thread_id_x]);
}

__global__ void log2_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=log2f(a[thread_id_x]);
}

__global__ void log10_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=log10f(a[thread_id_x]);
}

__global__ void logn_(float* a, long int size, float n){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=logf(a[thread_id_x])/logf(n);
}

__global__ void mod_(float* a, long int size, float v){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=fmodf(a[thread_id_x], v);
}

__global__ void inv_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x] = 1.0f/a[thread_id_x];
}

__global__ void mult_(float* a, long int size, float v){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x] *= v;
}

__global__ void normalize_(float* a, long int size, float min_ori, float max_ori, float min, float max){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=(max-min)/(max_ori-min_ori) * (a[thread_id_x]-min_ori) + min;
}

__global__ void pow_(float* a, long int size, float exp){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=powf(a[thread_id_x], exp);
}

__global__ void reciprocal_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=1.0f/a[thread_id_x];
}

__global__ void remainder_(float* a, long int size, float v){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x] = (int)(a[thread_id_x]/v);
}

__global__ void round_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=roundf(a[thread_id_x]);
}

__global__ void rsqrt_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=1.0f/sqrtf(a[thread_id_x]);
}

__global__ void sigmoid_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x] = expf(a[thread_id_x])/(expf(a[thread_id_x])+1.0f);
}

__global__ void sign_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        if(a[thread_id_x] > 0.0f){
            a[thread_id_x] = 1.0f;
        }else if(a[thread_id_x] < 0.0f){
            a[thread_id_x] = -1.0f;
        }else{
            a[thread_id_x] = 0.0f;
        }
    }
}

__global__ void sin_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=sinf(a[thread_id_x]);
}

__global__ void sinh_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=sinhf(a[thread_id_x]);
}

__global__ void sqr_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]*=a[thread_id_x];
}

__global__ void sqrt_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=sqrtf(a[thread_id_x]);
}

__global__ void tan_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=tanf(a[thread_id_x]);
}

__global__ void tanh_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]=tanhf(a[thread_id_x]);
}

__global__ void trunc_(float* a, long int size){

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        a[thread_id_x]= truncf(a[thread_id_x]);
}



///////////////////////////////////////////

__global__ void addc(float scA,float* a,float scB,float *b, float *c,long int incC, long int size)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size) {
        if (incC) c[thread_id_x]+=scA*a[thread_id_x]+scB*b[thread_id_x];
        else c[thread_id_x]=scA*a[thread_id_x]+scB*b[thread_id_x];
    }
}

__global__ void el_mult(float* a, float *b, float *c, long int incC, long int size)
{

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        if (incC) c[thread_id_x]+=a[thread_id_x]*b[thread_id_x];
        else c[thread_id_x]=a[thread_id_x]*b[thread_id_x];
}

__global__ void el_div(float* a, float *b, float *c, long int incC, long int size)
{

    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size)
        if (incC) c[thread_id_x]+=a[thread_id_x]/(b[thread_id_x]);
        else c[thread_id_x]=a[thread_id_x]/(b[thread_id_x]);
}


__global__ void sum_mat_row(float* a, float* b, float* c, long int rows,long int cols)
{
  long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        c[thread_id_x]=a[thread_id_x]+b[thread_id_x%cols];

}

__global__ void sum_mat_col(float* a, float* b, float* c, long int rows,long int cols)
{
  long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops)
        c[thread_id_x]=a[thread_id_x]+b[thread_id_x/cols];

}


__global__ void reduce_sum2D(float *a,float *b,long int rows,long int cols,long int axis)
{
  long int ops=rows*cols;
  long int thread_id_x = threadIdx.x+(blockDim.x*blockIdx.x);

  if (thread_id_x < ops){
    if (axis==0)
      atomicAdd(&(b[thread_id_x%cols]),a[thread_id_x]);
    else
      atomicAdd(&(b[thread_id_x/cols]),a[thread_id_x]);
  }

}

__global__ void reduceToSum(float *a, float *b, int a_ndim){
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+(blockDim.x*blockIdx.x);

    if (thread_id_x < ops){
        for (int j = 0; j < a_ndim; j++)
            b[thread_id_x] += a[j];
    }

}
