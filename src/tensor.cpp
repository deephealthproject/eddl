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
#include <vector>

#include "tensor.h"

#ifdef useGPU
#include "gpu/tensor_cuda.h"
#endif

#include "cpu/Eigen/Dense"

using namespace Eigen;


// Tensor class
Tensor::Tensor():device(0),dim(0),tam(0){}

Tensor::Tensor(const std::initializer_list<int>& init):Tensor(init,0){}
Tensor::Tensor(const std::initializer_list<int>& init, int dev):Tensor(shape(init.begin(), init.end()),dev){}

Tensor::Tensor(const shape s):Tensor(s,0){}
//Tensor::Tensor(const shape s, int dev):Tensor(s.size(),&((std::vector<int>(s.begin(), s.end()))[0]),dev){}

Tensor::Tensor(shape s,int dev)
{
  device=dev;
  dim=s.size();
  tam=1;
  sizes=s;

  for(int i=0;i<dim;++i) {
      tam*=s[i];
  }

  if (dev==DEV_CPU) {
    if (dim==1) ptr1.resize(sizes[0]);
    if (dim==2) ptr2.resize(sizes[0],sizes[1]);
    else {
      ptr=(Tensor **)malloc(sizes[0]*sizeof(Tensor *));
      s.erase(s.begin());
      for(int i=0;i<sizes[0];++i)
        ptr[i]=new Tensor(s,dev);
    }
  }
  #ifdef useGPU
  else if (device==DEV_GPU) g_ptr=create_tensor(tam);
  #endif
}

///////////////////////////////////////////
Tensor::~Tensor()
{
  if (device==DEV_CPU) {
    if (dim==1) ptr1.resize(0);
    else if (dim==2) ptr2.resize(0,0);
    else if (dim>2) {
      for(int i=0;i<sizes[0];++i) {
        delete ptr[i];
      }
      delete ptr;
    }
  }
}

///////////////////////////////////////////
shape Tensor::getshape()
{
  shape s=sizes;
  return s;
}

void Tensor::info()
{
  int i;

  fprintf(stderr,"DIM=%d\n",dim);
  fprintf(stderr,"(");
  for (i = 0; i < dim-1; i++)
		fprintf(stderr,"%d,",sizes[i]);
  fprintf(stderr,"%d)\n",sizes[i]);

  if (device==DEV_CPU) fprintf(stderr,"Device=CPU\n");
  else if (device==DEV_GPU) fprintf(stderr,"Device=GPU\n");
  else fprintf(stderr,"Device=FPGA\n");
  fprintf(stderr,"\n");
}

///////////////////////////////////////////
int Tensor::eqsize(Tensor *A, Tensor *B) {
  if (A->dim!=B->dim) return 0;

  for(int i=0;i<A->dim;i++)
    if (A->sizes[i]!=B->sizes[i]) return 0;

  return 1;

}
