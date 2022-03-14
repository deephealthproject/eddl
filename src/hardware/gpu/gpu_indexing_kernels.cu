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

__global__ void gpu_where_back(float *condition, float *PD_A, float *PD_B, float *D, long int size){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        if((bool) condition[thread_id_x]){
            PD_A[thread_id_x] += D[thread_id_x];
        }else{
            PD_B[thread_id_x] += D[thread_id_x];
        }
    }
}