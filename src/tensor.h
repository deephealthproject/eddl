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

#define INT32 1
#define FLOAT32 3
#define FLOAT64 4



#include <initializer_list>
#include <vector>
#include <string>

#include "cpu/Eigen/Dense"
#ifdef cGPU
#include "gpu/tensor_cuda.h"
#endif

using namespace std;


typedef vector<int> shape;

void msg(string s);
void msg(string s,string s2);

//using namespace Eigen;

class Tensor{

  public:
  int device;
  int dim;
  int tam;
  int type;
  shape sizes;

  Tensor **ptr;

  // CPU
  Eigen::RowVectorXd ptr1d;
  Eigen::RowVectorXf ptr1f;
  Eigen::RowVectorXi ptr1i;

  Eigen::MatrixXd ptr2d;
  Eigen::MatrixXf ptr2f;
  Eigen::MatrixXi ptr2i;
  ////

  // GPU
  float *gptrf;
  int *gptri;
  double *gptrd;  //

  Tensor();
  ~Tensor();

  Tensor(const initializer_list<int>& init);
  Tensor(const initializer_list<int>& init, int dev);
  Tensor(const initializer_list<int>& init, string type);
  Tensor(const initializer_list<int>& init, int dev,string type);

  Tensor(const shape s);
  Tensor(const shape s, int dev);
  Tensor(const shape s, string t);
  Tensor(const shape s, int dev, string t);

  shape getshape();
  void info();
  void changetype(int t);
  Tensor *clone();
  void mem(int t);
  void clean(int t);
  void print();
  void rand();
  //Tensor(Tensor &A);
  /////////

  static int eqsize(Tensor *A, Tensor *B);
  static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C);
  static void sum2D(Tensor *A, Tensor *B, Tensor *C,int incC);
  static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
  static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);

};



#endif
