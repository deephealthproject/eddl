// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
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

#include <stdio.h>
#include <stdio.h>
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
void Tensor::set_log() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = log(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::set_log2() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = log2f(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        //gpu_log(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::set_log10() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = log10f(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        //gpu_log(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::set_abs() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = fabs(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        //gpu_abs(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::set_exp() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = exp(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_exp(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

///////////////////////////////////////////
void Tensor::set_sqrt() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = sqrt(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_sqrt(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::set_sqr() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] *= ptr[i];
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_sqr(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////
float Tensor::total_sum() {

    if (isCPU()) {
        float sum = 0.0;

        for (int i = 0; i < size; ++i) sum += ptr[i];

        return sum;
    }
#ifdef cGPU
    else if (isGPU())
      {
         float sum;
         gpu_total_sum(this,&sum);
         return sum;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    return 0;
}


///////////////////////////////////////
float Tensor::total_abs() {

    if (isCPU()) {
        float sum = 0.0;

        for (int i = 0; i < size; ++i) sum += fabs(ptr[i]);

        return sum;
    }
#ifdef cGPU
    else if (isGPU())
      {
         float sum;
         gpu_total_sum(this,&sum);
         return sum;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    return 0;
}

