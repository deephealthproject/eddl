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
//// Dimensions and types must be compatible
//// Only for 2D Tensors
///////////////////////////////////////
void Tensor::mult2D(Tensor *A, Tensor *B, Tensor *C)
{
  int aux=0;

  if ((A->type!=B->type)||(A->type!=C->type)) msg("Incompatible types in mult2D");
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in mult2D");
  if ((A->dim!=2)||(B->dim!=2)||(C->dim!=2)) msg("mult2D only for 2D tensors");
  if ((A->sizes[1]!=B->sizes[0])||(A->sizes[0]!=C->sizes[0])||(B->sizes[1]!=C->sizes[1])) msg("Incompatible dims in mult2D");

  if (A->device==DEV_CPU) {
    if (A->type==FLOAT32) C->ptr2f=A->ptr2f*B->ptr2f;
    if (A->type==FLOAT64) C->ptr2d=A->ptr2d*B->ptr2d;
    if (A->type==INT32) C->ptr2i=A->ptr2i*B->ptr2i;
  }

  #ifdef cGPU
  else
  {

  }
  #endif
}

///////////////////////////////////////
//// SUM2D C=A+B
//// Dimensions and types must be compatible
//// Only for 2D Tensors
///////////////////////////////////////
void Tensor::sum2D(Tensor *A, Tensor *B, Tensor *C)
{
  int aux=0;

  if ((A->type!=B->type)||(A->type!=C->type)) msg("Incompatible types in sum2D");
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in sum2D");
  if ((A->dim!=2)||(B->dim!=2)||(C->dim!=2)) msg("sum2D only for 2D tensors");
  if ((!eqsize(A,B))||(!eqsize(A,C))) msg("Incompatible dims in sum2D");

  if (A->device==DEV_CPU) {
    if (A->type==FLOAT32) C->ptr2f=A->ptr2f+B->ptr2f;
    if (A->type==FLOAT64) C->ptr2d=A->ptr2d+B->ptr2d;
    if (A->type==INT32) C->ptr2i=A->ptr2i+B->ptr2i;
  }
  #ifdef cGPU
  else
  {

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
  if ((A->type!=B->type)||(A->type!=C->type)) msg("Incompatible types in sum2D_rowwise");
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in sum2D_rowwise");
  if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_rowwise dims");
  if ((!eqsize(A,C))||(A->sizes[1]!=B->sizes[0])) msg("Incompatible dims in sum2D_rowwise");

  if (A->device==DEV_CPU) {
    if (A->type==FLOAT32) C->ptr2f=A->ptr2f.rowwise()+B->ptr1f;
    if (A->type==FLOAT64) C->ptr2d=A->ptr2d.rowwise()+B->ptr1d;
    if (A->type==INT32) C->ptr2i=A->ptr2i.rowwise()+B->ptr1i;
  }
  #ifdef cGPU
  else
  {

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
  if ((A->type!=B->type)||(A->type!=C->type)) msg("Incompatible types in sum2D_colwise");
  if ((A->device!=B->device)||(A->device!=C->device)) msg("Tensors in different devices in sum2D_colwise");
  if ((A->dim!=2)||(B->dim!=1)||(C->dim!=2)) msg("sum2D_colwise dims");
  if ((!eqsize(A,C))||(A->sizes[0]!=B->sizes[0])) msg("Incompatible dims in sum2D_colwise");

  if (A->device==DEV_CPU) {
    if (A->type==FLOAT32) C->ptr2f=A->ptr2f.colwise()+B->ptr1f.transpose();
    if (A->type==FLOAT64) C->ptr2d=A->ptr2d.colwise()+B->ptr1d.transpose();
    if (A->type==INT32) C->ptr2i=A->ptr2i.colwise()+B->ptr1i.transpose();
  }
  #ifdef cGPU
  else
  {

  }
  #endif
}


///////////////////////////////////////////
///////////////////////////////////////////
///////////////////////////////////////////































//////
