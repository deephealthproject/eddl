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


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "tensor.h"

// only if gpu
#include "gpu/tensor_cuda.h"

extern int EDDL_DEV;

Tensor::Tensor()
{
  device=EDDL_DEV;
  dim=0;
}

Tensor::Tensor(int d1)
{
  device=EDDL_DEV;
  dim=1;
  tam=d1;
  size[0]=d1;

  if (device==DEV_CPU) ptr1.resize(d1);
  else if (device==DEV_GPU) g_ptr=create_tensor(tam);
}

Tensor::Tensor(int d1,int d2)
{
  device=EDDL_DEV;
  dim=2;
  tam=d1*d2;
  size[0]=d1;
  size[1]=d2;

  if (device==DEV_CPU) ptr2.resize(d1,d2);
  else if (device==DEV_GPU) g_ptr=create_tensor(tam);
}

Tensor::Tensor(int d1,int d2,int d3)
{
  device=EDDL_DEV;
  dim=3;
  tam=d1*d2*d3;
  size[0]=d1;
  size[1]=d2;
  size[2]=d3;

  if (device==DEV_CPU) {
    ptr=(Tensor **)malloc(d1*sizeof(Tensor *));
    for(int i=0;i<d1;++i)
      ptr[i]=new Tensor(d2,d3);
  }
  else if (device==DEV_GPU) g_ptr=create_tensor(tam);

}

Tensor::Tensor(int d1,int d2,int d3,int d4)
{
  device=EDDL_DEV;
  dim=4;
  tam=d1*d2*d3*d4;
  size[0]=d1;
  size[1]=d2;
  size[2]=d3;
  size[3]=d4;

  if (device==DEV_CPU) {
    ptr=(Tensor **)malloc(d1*sizeof(Tensor *));
    for(int i=0;i<d1;++i)
      ptr[i]=new Tensor(d2,d3,d4);
    }
  else if (device==DEV_GPU) g_ptr=create_tensor(tam);

}

Tensor::Tensor(int d1,int d2,int d3,int d4,int d5)
{
  device=EDDL_DEV;
  dim=5;
  tam=d1*d2*d3*d4*d5;
  size[0]=d1;
  size[1]=d2;
  size[2]=d3;
  size[3]=d4;
  size[4]=d5;

  if (device==DEV_CPU) {
    ptr=(Tensor **)malloc(d1*sizeof(Tensor *));
    for(int i=0;i<d1;++i)
      ptr[i]=new Tensor(d2,d3,d4,d5);
  }
  else if (device==DEV_GPU) g_ptr=create_tensor(tam);
}


///////////////////////////////////////////
Tensor::~Tensor()
{
  if (dim==1) ptr1.resize(0);
  else if (dim==2) ptr2.resize(0,0);
  else if (dim>2) {
    for(int i=0;i<size[0];++i) {
      delete ptr[i];
    }
    delete ptr;
  }
}


///////////////////////////////////////////
int Tensor::eqsize(Tensor *A, Tensor *B) {
  if (A->dim!=B->dim) return 0;

  for(int i=0;i<A->dim;i++)
    if (A->size[i]!=B->size[i]) return 0;

  return 1;

}
