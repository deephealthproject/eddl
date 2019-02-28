// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _TENSOR_CUDA_KERNELS_
#define _TENSOR_CUDA_KERNELS_

#include <cuda.h>

__global__ void fill(float *aptr,float *bptr,int t,int aini,int at,int bini,int bt,int tot,int inc);

__global__ void set(float* a, float v, long int rows, long int cols);
__global__ void mult(float* a, float v, long int rows, long int cols);
__global__ void sum(float* a, float v, long int rows, long int cols);


__global__ void log(float* a, long int rows, long int cols);
__global__ void exp(float* a, long int rows, long int cols);
__global__ void sqrt(float* a,long int rows, long int cols);
__global__ void sqr(float* a, long int rows, long int cols);
__global__ void mask(float* a, float v, long int rows, long int cols);

__global__ void reduce_array_sum(float* array, long int ops, long int cols,float* result);

__global__ void sum(float scA,float* a,float scB,float *b, float *c,long int incC, long int tam);
__global__ void el_mult(float* a,float *b, float *c, long int incC, long int rows, long int cols);
__global__ void el_div(float* a, float *b, float *c, long int incC, long int rows, long int cols);

__global__ void sum_mat_row(float* a, float* b, float* c, long int cols, long int rows);
__global__ void sum_mat_col(float* a, float* b, float* c, long int cols, long int rows);

__global__ void reduce_sum2D(float *a,float *b,long int r,long int c,long int axis);

__global__ void cent(float* a, float* b, float* c, long int tam);
__global__ void accuracy(float* T, float* N,float* acc,long int cols, long int total_ops, int* MC_err);

__global__ void relu(float *a,float *b,long int tam);
__global__ void d_relu(float *d,float *i,float *pd,long int tam);
__global__ void softmax(float* E,float* N,float* auxE ,long int sample_dim, long int n_vals);

#endif





















//////
