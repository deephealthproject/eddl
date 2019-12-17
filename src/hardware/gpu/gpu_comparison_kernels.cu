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

__global__  void gpu_allclose(float *A, float *B, float rtol, float atol, bool equal_nan, int size, bool &allclose){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        bool close = fabsf(A[thread_id_x] - B[thread_id_x]) <= (atol + rtol * fabsf(B[thread_id_x]));
        if (!close){
            allclose = false;
            return;
        }
    }
}

__global__  void gpu_isclose(float *A, float *B, float *C, float rtol, float atol, bool equal_nan, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = fabsf(A[thread_id_x] - B[thread_id_x]) <= (atol + rtol * fabsf(B[thread_id_x]));
    }
}

__global__  void gpu_greater(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] > B[thread_id_x];
    }
}

__global__  void gpu_greater_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] >= B[thread_id_x];
    }
}

__global__  void gpu_less(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] < B[thread_id_x];
    }
}

__global__  void gpu_less_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] <= B[thread_id_x];
    }
}

__global__  void gpu_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] == B[thread_id_x];
    }
}

__global__  void gpu_not_equal(float *A, float *B, float *C, int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        C[thread_id_x] = A[thread_id_x] != B[thread_id_x];
    }
}
