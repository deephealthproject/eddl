/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
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

__global__ void range(float* a, float start, float step, long int size) {
    long int thread_id_x = blockDim.x*blockIdx.x + threadIdx.x;

    if (thread_id_x < size)
        a[thread_id_x]= start + step*(float)(thread_id_x);
}


__global__ void eye(float* a, long int rows, long int cols, int offset) {
    long int ops = rows*cols;
    long int thread_id_x = blockDim.x*blockIdx.x + threadIdx.x;

    if (thread_id_x < ops)
        if ((thread_id_x/rows + offset) == (thread_id_x%cols)){ a[thread_id_x] = 1.0f; }
        else { a[thread_id_x] = 0.0f; }
}


__global__ void gpu_diag(float* A, float* B, long int rows, long int cols, int k) {
    long int ops = rows*cols;
    long int thread_id_x = blockDim.x*blockIdx.x + threadIdx.x;

    if (thread_id_x < ops)
        if ((thread_id_x/rows + k) == (thread_id_x%cols)){ B[thread_id_x] = A[thread_id_x]; }
        else { B[thread_id_x] = 0.0f; }
}