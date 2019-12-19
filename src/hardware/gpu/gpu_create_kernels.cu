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
