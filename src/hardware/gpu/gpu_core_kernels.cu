/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
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

__global__ void select_rows(float* A, float* B, int rowsize, int size, int* indices, int ini, bool mask_zeros)
{
  long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id_x < size){
    int b=thread_id_x/rowsize;
    int c=thread_id_x%rowsize;

    int posA=(indices[b]*rowsize)+c;
    int posB=((b-ini)*rowsize)+c;

    if ((mask_zeros)&&(indices[b]==0)) B[posB]=0;
    else B[posB]=A[posA];
  }

}
__global__ void deselect_rows(float* A, float* B, int rowsize, int size, int* indices, int ini, int inc, bool mask_zeros)
{
  long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id_x < size){
    int b=thread_id_x/rowsize;
    int c=thread_id_x%rowsize;

    int posA=((b-ini)*rowsize)+c;
    int posB=(indices[b]*rowsize)+c;

    if ((mask_zeros)&&(indices[b]==0)) B[posB]=0;
    else {
      if (inc) B[posB]+=A[posA];
      else B[posB]=A[posA];
    }
  }

}

__global__ void select(float* A, float* B, long int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = A[indices[thread_id_x]];
    }
}

__global__ void select_back(float* A, float* B, long int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[indices[thread_id_x]] += A[thread_id_x];
    }
}

__global__ void set_select(float* A, float* B, long int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        A[indices[thread_id_x]] = B[thread_id_x];
    }
}

__global__ void set_select_back(float* A, float* B, long int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] += A[indices[thread_id_x]];
    }
}


__global__ void gpu_gather(float* A, float* B, long int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        A[indices[thread_id_x]] = B[thread_id_x];
    }
}


__global__ void gpu_expand(float* A, float* B, long int size, int* indices){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        B[thread_id_x] = A[indices[thread_id_x]];
    }
}


__global__ void gpu_repeat(float *A, float* B, unsigned int* repeats, unsigned int axis,
                           long int A_size, unsigned int* A_shape, unsigned int* A_strides,  unsigned int* B_strides, unsigned int ndim, bool derivative){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < A_size){
        auto* A_indices = new unsigned int[ndim];
        auto* B_indices = new unsigned int[ndim];

        // Get A Indices
        // Get B indices. Same as A indices but changing size in axis to be expanded
        for(int i=0; i<ndim; i++) {
            A_indices[i] = thread_id_x / A_strides[i] % A_shape[i];
            B_indices[i] = A_indices[i];
        }

        // Get A_indices[axis]=> repeat(3,2,1) AND "sel_index=2" => start at position: 3+2=5
        unsigned int A_idx_axis = A_indices[axis]; // (2, 0) => axis=0 => 2
        unsigned int B_idx_axis = 0;
        for (unsigned int j = 0; j < A_idx_axis; j++) { B_idx_axis += repeats[j]; }
        B_indices[axis] = B_idx_axis;

        // Get address
        unsigned int B_address = 0;
        for (int i=0; i< ndim; i++){
            B_address += B_indices[i] * B_strides[i];
        }

        // Copy value t times
        for (unsigned int t = 0; t < repeats[A_indices[axis]]; t++) {
            if (!derivative){
                B[B_address + t*B_strides[axis]] = A[thread_id_x];
            }else{
                A[thread_id_x] += B[B_address + t*B_strides[axis]];
            }
        }

        // Delete stuff
        delete[] A_indices;
        delete[] B_indices;
    }
}


__global__ void gpu_repeat_batch(float* A, float* B, long int A_size, long int B_size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < B_size){
        B[thread_id_x] = A[thread_id_x%A_size];
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
