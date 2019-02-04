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
#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>
#include "tensor.h"

#ifdef cGPU
#include "gpu/tensor_cuda.h"
#include "gpu/tensor_cuda_op.h"
#endif


using namespace std;


///////////////////////////////////////////
/// TENSOR OPERATIONS AS STATIC METHODS ///
///////////////////////////////////////////

int Tensor::eqsize(Tensor *A, Tensor *B) {
  if (A->dim!=B->dim) return 0;

  for(int i=0;i<A->dim;i++)
    if (A->sizes[i]!=B->sizes[i]) return 0;

  return 1;
}



///////////////////////////////////////
//// MULT2D C=A*B
//// tA means transpose A {0,1}
//// tB means transpose B {0,1}
//// tC 1 means C+=A*B (increment over C)
//// Dimensions and types must be compatible
//// Only for 2D Tensors
///////////////////////////////////////
void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC)
{

  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in mult2D");
  if ((A->dim!=2)||(B->dim!=2)||(C->dim!=2)) msg("mult2D only for 2D tensors");
  if (!tA) {
  if (!tB) {
      if ((A->sizes[1]!=B->sizes[0])||(A->sizes[0]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims in mult2D");
    }
  else
    if ((A->sizes[1]!=B->sizes[1])||(A->sizes[0]!=C->sizes[0])||(B->sizes[0]!=C->sizes[1])) msg("Incompatible dims in mult2D");
  }
  else {
    if (!tB) {
        if ((A->sizes[0]!=B->sizes[0])||(A->sizes[1]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims in mult2D");
      }
    else
      if ((A->sizes[0]!=B->sizes[1])||(A->sizes[1]!=C->sizes[0])||(B->sizes[0]!=C->sizes[1])) msg("Incompatible dims in mult2D");
  }

  if (A->device==DEV_CPU) {
      if (!tB) {
        if (!tA){
          if (!incC) C->ptr2=A->ptr2*B->ptr2;
          else C->ptr2+=A->ptr2*B->ptr2;
        }
        else {
          if (!incC) C->ptr2=A->ptr2.transpose()*B->ptr2;
          else C->ptr2+=A->ptr2.transpose()*B->ptr2;
        }
      }
      else {
        if (!tA){
          if (!incC) C->ptr2=A->ptr2*B->ptr2.transpose();
          else C->ptr2+=A->ptr2*B->ptr2.transpose();
        }
        else {
          if (!incC) C->ptr2=A->ptr2.transpose()*B->ptr2.transpose();
          else C->ptr2+=A->ptr2.transpose()*B->ptr2.transpose();
        }
      }
    }
  #ifdef cGPU
  else if (A->device<DEV_FPGA)
  {
    gpu_mult2D(A,tA,B,tB,C,incC);
  }
  #endif
}

///////////////////////////////////////
//// SUM2D C=A+B
//// or C+=A+B if incC is 1
//// Dimensions and types must be compatible
//// Only for 2D Tensors
///////////////////////////////////////
void Tensor::sum2D(Tensor *A, Tensor *B, Tensor *C,int incC)
{
  int aux=0;

  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in sum2D");
  if ((A->dim!=2)||(B->dim!=2)||(C->dim!=2)) msg("sum2D only for 2D tensors");
  if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims in sum2D");

  if (A->device==DEV_CPU) {
      if (incC) C->ptr2+=A->ptr2+B->ptr2;
      else C->ptr2=A->ptr2+B->ptr2;
  }
  #ifdef cGPU
  else if (A->device<DEV_FPGA)
  {
     gpu_sum2D(A,B,C,incC);
  }
  #endif
}

///////////////////////////////////////
//// SUM2D_rowise C=A.rowise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C)
{
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in sum2D_rowwise");
  if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_rowwise dims");
  if ((!eqsize(A,C))||(A->sizes[1]!=B->sizes[0])) msg("Incompatible dims in sum2D_rowwise");


  if (A->device==DEV_CPU) C->ptr2=A->ptr2.rowwise()+B->ptr1;
  #ifdef cGPU
  else if (A->device<DEV_FPGA)
  {
    gpu_sum2D_rowwise(A,B,C);

  }
  #endif
}

///////////////////////////////////////
//// SUM2D_colwise C=A.colwise+B
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
///////////////////////////////////////
void Tensor::sum2D_colwise(Tensor *A, Tensor *B, Tensor *C)
{
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in sum2D_colwise");
  if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_colwise dims");
  if ((!eqsize(A,C))||(A->sizes[0]!=B->sizes[0])) msg("Incompatible dims in sum2D_colwise");

  if (A->device==DEV_CPU) C->ptr2=A->ptr2.colwise()+B->ptr1.transpose();
  #ifdef cGPU
  else if (A->device<DEV_FPGA)
  {
    gpu_sum2D_colwise(A,B,C);
  }
  #endif
}

///////////////////////////////////////
//// reduce_sum2D B=reduce_sum(A)
//// Dimensions and types must be compatible
//// A is 2D Tensor
//// B is 1D Tensor
//// axis is the dimension to be sumed
///////////////////////////////////////
void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis,int incB)
{
  if (A->device!=B->device) msg("Tensors in different devices in reduce_sum2D");
  if ((A->dim-1)!=B->dim) msg("Incorrect dims in reduce_sum2D");
  if ((A->sizes[1-axis]!=B->sizes[0])) msg("Incompatible dims in reduce_sum2D");

  if (A->device==DEV_CPU) {
    if (axis=0) {
      #pragma omp parallel for
      for(int i=0;i<A->sizes[1];++i) {
        if (!incB) B->ptr1(i)=0;
        for(int j=0;j<A->sizes[0];++j)
           B->ptr1(i)+=A->ptr2(j,i);
      }
    }
    else {
     #pragma omp parallel for
     for(int i=0;i<A->sizes[0];++i) {
        if (!incB) B->ptr1(i)=0;
        for(int j=0;j<A->sizes[1];++j)
          B->ptr1(i)+=A->ptr2(i,j);
      }
    }
  }
  #ifdef cGPU
  else if (A->device<DEV_FPGA)
  {
    gpu_reduce_sum2D(A,B,axis,incB);
  }
  #endif
}

///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////































//////
