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
#define DEV_GPU 1000
#define DEV_FPGA 2000


#include <initializer_list>
#include <vector>
#include <string>


#include "cpu/Eigen/Dense"

#define MAX_GPUS 8

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
  shape sizes;

  Tensor **ptr;

  // CPU
  Eigen::RowVectorXf ptr1;
  Eigen::MatrixXf ptr2;


  // GPU
  float *gptr;
  int gpu_device;

  //FPGA


  // Constructors
  Tensor();
  Tensor(const initializer_list<int>& init);
  Tensor(const initializer_list<int>& init, int dev);

  Tensor(const shape s);
  Tensor(const shape s, int dev);

  ~Tensor();
  ///////// normal metods
  shape getshape();
  void info();
  Tensor *clone();
  void print();
  void rand();
  void set(float v);


  ///////// static metods
  static int eqsize(Tensor *A, Tensor *B);
  static void mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC);
  static void sum2D(Tensor *A, Tensor *B, Tensor *C,int incC);
  static void sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C);
  static void sum2D_colwise(Tensor *A, Tensor *B, Tensor *C);






};





#endif
