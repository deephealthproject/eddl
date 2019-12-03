/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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


__global__ void fill_(float* a, float v, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        a[thread_id_x]=v;
    }
}

__global__ void fill(float *aptr,float *bptr,int t,int aini,int at,int bini,int bt,int tot,int inc){
    int i=blockIdx.x;
    int j=threadIdx.x;
    int k=blockIdx.y;

    int ap=(i*at)+((aini+j)*t)+k;
    int bp=(i*bt)+((bini+j)*t)+k;

    if (bp<tot){
        if (inc) {
            bptr[bp] += aptr[ap];
        } else { bptr[bp]=aptr[ap];}
    }

}


__global__ void mask(float* a, float v, long int size){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        a[thread_id_x]=a[thread_id_x]<v;
    }
}


__global__ void select(float* A, float* B, int batch, int* A_stride, int* B_stride, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    long int size = batch * B_stride[0];  // B is the small

    if (thread_id_x < size){
        int b = thread_id_x / B_stride[0] % batch;

        int A_str_batch = b * A_stride[0];
        int B_str_batch = b * B_stride[0];

        int i = thread_id_x % B_stride[0];
        int A_pos = A_str_batch + indices[i];
        int B_pos = B_str_batch + i;

        B[B_pos] = A[A_pos];
    }
}

__global__ void select_back(float* A, float* B, int batch, int* A_stride, int* B_stride, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;
    long int size = batch * A_stride[0];  // A is the small

    if (thread_id_x < size){
        int b = thread_id_x / A_stride[0] % batch;

        int A_str_batch = b * A_stride[0];
        int B_str_batch = b * B_stride[0];

        int i = thread_id_x % A_stride[0];
        int A_pos = A_str_batch + i;
        int B_pos = B_str_batch + indices[i];

        B[B_pos] += A[A_pos];
    }
}