/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/gpu_kernels.h"


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


__global__ void select(float* A, float* B, int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = A[indices[thread_id_x]];
    }
}

__global__ void select_back(float* A, float* B, int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[indices[thread_id_x]] += A[thread_id_x];
    }
}

__global__ void set_select(float* A, float* B, int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        A[indices[thread_id_x]] = B[thread_id_x];
    }
}

__global__ void set_select_back(float* A, float* B, int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] += A[indices[thread_id_x]];
    }
}



__global__ void concat(float *dest, float *src, unsigned int src_size, unsigned int src_stride, unsigned int dest_stride, bool derivative){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < src_size){
        unsigned int k = thread_id_x % src_stride;  // Pos (index) in the stride (src)
        unsigned int stride_idx = thread_id_x / src_stride;  // Index of the stride (src/dst)
        unsigned int dest_offset = stride_idx * dest_stride;  // Offset in dest

        if(derivative){ src[thread_id_x] += dest[dest_offset + k]; }
        else{ dest[dest_offset + k] = src[thread_id_x]; }
    }


}
