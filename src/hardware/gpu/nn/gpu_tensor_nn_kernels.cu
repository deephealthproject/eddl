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



__global__ void repeat_nn_k(float *a, int batch, int depth, int a_rows, int a_cols, float *b, int b_rows, int b_cols, int *size){
    long int ops=batch*depth*b_rows*b_cols;
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x;

    if (thread_id_x < ops){
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int b_rcd=depth*b_rows*b_cols; // out size of batch
        int b_rc=b_rows*b_cols;  // out size of slice

        int batch_i=thread_id_x/b_rcd; // current batch (ib=ob)
        int bm=thread_id_x%b_rcd; // index in batch
        int depth_i=bm/b_rc; // current batch (ib=ob)

        int row_b=(bm%b_rc)/b_cols; // out row
        int col_b=(bm%b_rc)%b_cols; // out col

        int a_rcd=depth*a_rows*a_cols; // out size of batch
        int a_rc=a_rows*a_cols;  // out size of slice

        int arow_i = row_b/size[0];
        int acol_i = col_b/size[1];
        long int offset_a = (batch_i*a_rcd) + (depth_i*a_rc) + (arow_i*a_cols) + acol_i;
//        printf("offset_a: %ld, batch: %d, depth: %d, arow_i: %d, acol_i: %d\n", offset_a, batch_i, depth_i, arow_i, acol_i );

        b[thread_id_x] = a[offset_a];
    }
}

__global__ void d_repeat_nn_k(float *d, int batch, int depth, int d_rows, int d_cols, float *a, int a_rows, int a_cols, int *size){
    long int ops=batch*depth*d_rows*d_cols;
    long int thread_id_x = blockIdx.x*blockDim.x + threadIdx.x;

    if (thread_id_x < ops){
        // output pixel at batch=ob, coord=(or,oc) at map=oz
        int d_rcd=depth*d_rows*d_cols; // out size of batch
        int d_rc=d_rows*d_cols;  // out size of slice

        int batch_i=thread_id_x/d_rcd; // current batch (ib=ob)
        int bm=thread_id_x%d_rcd; // index in batch
        int depth_i=bm/d_rc; // current batch (ib=ob)

        int row_d=(bm%d_rc)/d_cols; // out row
        int col_d=(bm%d_rc)%d_cols; // out col

        int a_rcd=depth*a_rows*a_cols; // out size of batch
        int a_rc=a_rows*a_cols;  // out size of slice

        int arow_i = row_d/size[0];
        int acol_i = col_d/size[1];
        long int offset_a = (batch_i*a_rcd) + (depth_i*a_rc) + (arow_i*a_cols) + acol_i;
//        printf("offset_a: %ld, batch: %d, depth: %d, arow_i: %d, acol_i: %d\n", offset_a, batch_i, depth_i, arow_i, acol_i );

        atomicAdd(&(a[offset_a]), d[thread_id_x]);
    }
}


__global__ void gpu_select_nn(float* A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        long int b = thread_id_x / B_batch_str;
        long int i = thread_id_x % B_batch_str;
        B[thread_id_x] = A[b*A_batch_str + indices[i]];
    }
}

__global__ void gpu_select_back_nn(float* A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        long int b = thread_id_x / A_batch_str;
        long int i = thread_id_x % A_batch_str;
        B[b*B_batch_str + indices[i]] += A[thread_id_x];
    }
}

__global__ void gpu_set_select_nn(float* A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        int b = thread_id_x / B_batch_str;
        int i = thread_id_x % B_batch_str;
        A[b*A_batch_str + indices[i]] = B[thread_id_x];
    }
}

__global__ void gpu_set_select_back_nn(float* A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str){
    long int thread_id_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id_x < size){
        int b = thread_id_x / B_batch_str;
        int i = thread_id_x % B_batch_str;
        B[thread_id_x] += A[b*A_batch_str + indices[i]];
    }
}