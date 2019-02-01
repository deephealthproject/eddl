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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

///////////////////////////////////////////
__global__ void sum_mat_row(float* a, float* b, float* c, int rows, int cols)
{
 int ops=rows*cols;
 int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

 if (thread_id_x < ops)
   c[thread_id_x]=a[thread_id_x]+b[thread_id_x%cols];

}
///////////////////////////////////////////
__global__ void set(float* a, float v, int rows, int cols)
{
 int ops=rows*cols;
 int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
 printf("[%d %d %d %d]\n",threadIdx.x,blockIdx.x,blockDim.x,thread_id_x);

 if (thread_id_x < ops){
   a[thread_id_x]=v;
   printf("a[%d]=%f\n",thread_id_x,v);
 }

}

///////////////////////////////////////////


///////////////////////////////////////////


///////////////////////////////////////////


///////////////////////////////////////////


///////////////////////////////////////////
