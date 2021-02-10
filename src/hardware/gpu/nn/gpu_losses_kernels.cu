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

#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"


__global__ void cent(float* a, float* b, float* c, long int size)
{

 long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < size){
   c[thread_id_x]=0;
   if (a[thread_id_x]) c[thread_id_x]-=a[thread_id_x]*logf(b[thread_id_x]+0.00001);
   if (a[thread_id_x]!=1.0) c[thread_id_x]-=(1.0-a[thread_id_x])*logf(1.0-b[thread_id_x]+0.00001);
  }
}



__global__ void gpu_categorical_cross_entropy(float* y_true, float* y_pred, float* sum_array, unsigned int n_batches, unsigned int n_features){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Batch index

    if (thread_id_x < n_batches){
        float eps =10e-8;
        unsigned int batch_i = thread_id_x; // Alias

        // Contiguous data
        unsigned int start = batch_i*n_features;
        unsigned int end = start+n_features;

        // Compute cross-entropy
        float bi_sum = 0.0f;
        for (unsigned int i = start; i<end; i++) {
            bi_sum += y_true[i] * logf(y_pred[i]+eps);
        }

        // Store partial sums (later will be reduced)
        sum_array[thread_id_x] = bi_sum;
    }
}

__global__ void gpu_d_categorical_cross_entropy(float* y_true, float* y_pred, float* delta, long int size){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Index

    if (thread_id_x < size){
        float eps =10e-8;
        delta[thread_id_x] = -y_true[thread_id_x] * (1.0f/ (y_pred[thread_id_x]+eps) );
    }
}

__global__ void gpu_binary_cross_entropy(float* y_true, float* y_pred, float* sum_array, unsigned int size){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Index

    if (thread_id_x < size){
        float eps =10e-8;

        // Store sums (later will be reduced)
        sum_array[thread_id_x] = y_true[thread_id_x] * logf(y_pred[thread_id_x]+eps) + (1.0-y_true[thread_id_x]) * logf(1.0f-y_pred[thread_id_x]+eps);
    }

}

__global__ void gpu_d_binary_cross_entropy(float* y_true, float* y_pred, float* delta, long int size){
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x; // Index

    if (thread_id_x < size){
        float eps =10e-8;
        delta[thread_id_x] = -( y_true[thread_id_x] * 1.0f/(y_pred[thread_id_x]+eps) + (1.0-y_true[thread_id_x]) * 1.0f/(1.0f-y_pred[thread_id_x]+eps) * -1.0f );
    }
}

