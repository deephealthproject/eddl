// The MIT License (MIT)
//
// Copyright (c) 2019 PRHLT Research Group. Inc. http://www.prhlt.upv.es
//
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
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <initializer_list>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "../utils.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;


///////////////////////////////////////////
void Tensor::set(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_set(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::mult(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] *= v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_mult(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::div(float v) { mult(1.0 / v); }

///////////////////////////////////////////
void Tensor::sum(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] += v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_sum(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::sub(float v) { sum(-v); }


///////////////////////////////////////////
void Tensor::rand_uniform(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = uniform() * v;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_uniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


///////////////////////////////////////////
void Tensor::rand_suniform(float v) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = suniform() * v;

    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_suniform(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif


}


void Tensor::rand_binary(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i)
            if (uniform() < v) ptr[i] = 1.0;
            else ptr[i] = 0.0;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_rand_binary(this,v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

}


///////////////////////////////////////////
