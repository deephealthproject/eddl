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

#ifndef _TENSOR_
#define _TENSOR_

#define DEV_CPU 0
#define DEV_GPU 1
#define DEV_FPGA 2

#define INT16 ((short int)1)
#define INT32 ((int)1)
#define INT64 ((long int)1)
#define FLOAT32 ((float)1.0)
#define FLOAT64 ((double)1.0)

#include <initializer_list>
#include <vector>
#include <string>

#include "cpu/Eigen/Dense"
#ifdef cGPU
#include "gpu/tensor_cuda.h"
#endif

#define TENSOR auto


typedef std::vector<int> shape;

using namespace Eigen;

template <class T>
class Tensor{

  public:
  int device;
  int dim;
  int tam;
  shape sizes;

  Tensor **ptr;

  // CPU
  RowVectorXd ptr1;
  Matrix<T,Dynamic, Dynamic> ptr2;
  ////

  // GPU
  float *g_ptr;
  //

  Tensor();
  ~Tensor();
  //Tensor(int d,int s[],int dev);
  Tensor(const std::initializer_list<int>& init);
  Tensor(const std::initializer_list<int>& init, int dev);
  Tensor(const shape s);
  Tensor(const shape s, int dev);
  /////////
  shape getshape();
  void info();
  /////////
  static int eqsize(Tensor *A, Tensor *B);


};

// Tensor class
template <class T>
Tensor<T>::Tensor():device(0),dim(0),tam(0){}

template <class T>
Tensor<T>::Tensor(const std::initializer_list<int>& init):Tensor(init,0){}

template <class T>
Tensor<T>::Tensor(const std::initializer_list<int>& init, int dev):Tensor(shape(init.begin(), init.end()),dev){}

template <class T>
Tensor<T>::Tensor(const shape s):Tensor(s,0){}

template <class T>
Tensor<T>::Tensor(shape s,int dev)
{
  #ifndef cGPU
  if (dev==DEV_GPU){
    fprintf(stderr,"Not compiled for GPU\n");
    exit(0);
  }
  #endif
  #ifndef cFPGA
  if (dev==DEV_FPGA){
    fprintf(stderr,"Not compiled for FPGA\n");
    exit(0);
  }
  #endif


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
template <class T>Tensor<T>::~Tensor()
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
template <class T>
shape Tensor<T>::getshape()
{
  shape s=sizes;
  return s;
}

template <class T>
void Tensor<T>::info()
{
  int i;

  fprintf(stderr,"DIM=%d\n",dim);
  fprintf(stderr,"(");
  for (i = 0; i < dim-1; i++)
		fprintf(stderr,"%d,",sizes[i]);
  fprintf(stderr,"%d)\n",sizes[i]);
  fprintf(stderr,"Total bytes=%ld\n",tam*sizeof(T));

  if (device==DEV_CPU) fprintf(stderr,"Device=CPU\n");
  else if (device==DEV_GPU) fprintf(stderr,"Device=GPU\n");
  else fprintf(stderr,"Device=FPGA\n");
  fprintf(stderr,"\n");
}

///////////////////////////////////////////
template <class T>
int Tensor<T>::eqsize(Tensor *A, Tensor *B) {
  if (A->dim!=B->dim) return 0;

  for(int i=0;i<A->dim;i++)
    if (A->sizes[i]!=B->sizes[i]) return 0;

  return 1;

}

///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////

template <typename T>
Tensor<T> *createtensor(const std::initializer_list<int>& init, int dev,T val)
{
  return new Tensor<T>(init,dev);
}

template <typename T>
Tensor<T> *createtensor(const std::initializer_list<int>& init, T val)
{
  return new Tensor<T>(init,0);
}






#endif
