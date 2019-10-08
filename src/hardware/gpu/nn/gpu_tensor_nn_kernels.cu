/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Juan Maroñas: jmaronas@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_nn_kernels.h"
#include "../gpu_kernels.h"



__global__ void repeat_nn(float *a, long int a_rows, long int a_cols, float *b, long int b_rows, long int b_cols, float *s, int size){
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops){
        // Get row/col of Tensor B
        int row_b = thread_id_x/b_cols;
        int col_b = thread_id_x%b_cols;

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_b/size[0];
        int col_a = col_b/size[1];
        int offset_a = row_a*a_cols + col_a;

        B->ptr[thread_id_x] = A->ptr[offset_a];
    }
}

__global__ void d_repeat_nn(float *d, long int d_rows, long int d_cols, float *a, long int a_rows, long int a_cols, float *s, int size){
    long int ops=rows*cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops){
        // Get row/col of Tensor B2
        int row_d = thread_id_x/d_cols;
        int col_d = thread_id_x%d_cols;

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_d/size[0];
        int col_a = col_d/size[1];
        int offset_a = row_a*a_cols + col_a;

        A->ptr[offset_a] += D->ptr[thread_id_x];
    }
}