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



__global__ void repeat_nn(float *a, int a_rows, int a_cols, float *b, int b_rows, int b_cols, int *size){
    long int ops=b_rows*b_cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops){
        // Get row/col of Tensor B
        int row_b = thread_id_x/b_cols;
        int col_b = thread_id_x%b_cols;

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_b/size[0];
        int col_a = col_b/size[1];
        int offset_a = row_a*a_cols + col_a;

        b[thread_id_x] = a[offset_a];
    }
}

__global__ void d_repeat_nn(float *d, int d_rows, int d_cols, float *a, int a_rows, int a_cols, int *size){
    long int ops=d_rows*d_cols;
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < ops){
        // Get row/col of Tensor B2
        int row_d = thread_id_x/d_cols;
        int col_d = thread_id_x%d_cols;

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_d/size[0];
        int col_a = col_d/size[1];
        int offset_a = row_a*a_cols + col_a;

        a[offset_a] += d[thread_id_x];
    }
}