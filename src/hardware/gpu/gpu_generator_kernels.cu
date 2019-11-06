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
#include <curand_kernel.h>
#include <curand.h>

#include "gpu_kernels.h"


__global__ void uniform_array(float* array, int size, unsigned long seed) {
    long int thread_id_x = blockDim.x*blockIdx.x* + threadIdx.x;

    if (thread_id_x < size) {
        curandState state;
        curand_init(seed, thread_id_x, 0, &state);  // opt. => seed=clock64()
        array[thread_id_x] = curand_uniform(&state);
    }
}