/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

__global__ void gpu_where(float *condition, float *A, float *B, float *C, long int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        if((bool) condition[thread_id_x]){
            C[thread_id_x] = A[thread_id_x];
        }else{
            C[thread_id_x] = B[thread_id_x];
        }
    }
}
