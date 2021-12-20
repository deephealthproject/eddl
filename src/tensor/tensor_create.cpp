/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

using namespace std;

Tensor* raw_range(float start, float step, int size, int dev){
    auto t = new Tensor(vector<int>{size}, nullptr, dev);
    if (t->isCPU()) {
        cpu_range(t, start, step);
    }
#ifdef cGPU
    else if (t->isGPU())
      {
        gpu_range(t, start, step);
      }
#endif
    return t;
}


// ************************************************
// Creation ops ***********************************
// ************************************************

Tensor* Tensor::empty(const vector<int> &shape, int dev){
    return new Tensor(shape, dev);
}

Tensor* Tensor::empty_like(Tensor *A){
    return Tensor::empty(A->shape, A->device);
}

Tensor* Tensor::zeros(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_(0.0f);
    return t;
}

Tensor* Tensor::zeros_like(Tensor *A){
    return Tensor::zeros(A->shape, A->device);
}

Tensor* Tensor::ones(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_(1.0f);
    return t;
}

Tensor* Tensor::ones_like(Tensor *A){
    return Tensor::ones(A->shape, A->device);
}

Tensor* Tensor::full(const vector<int> &shape, float value, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_(value);
    return t;
}

Tensor* Tensor::full_like(Tensor *A, float value){
    return Tensor::full(A->shape, value, A->device);
}

Tensor* Tensor::arange(float start, float end, float step, int dev){
    if(step==0.0f){ step = 1.0f; }  // Trick to avoid division by zero

    // [1, 100)
    // Returns a 1-D tensor of size ceil(end - start) with values from start to end with step step.
    // Step is the gap between two values in the tensor.
    int size = ::ceilf((end-start)/step);
    return raw_range(start, step, size, dev);
}

Tensor* Tensor::range(float start, float end, float step, int dev){
    if(step==0.0f){ step = 1.0f; }  // Trick to avoid division by zero

    // [1, 100]
    // Returns a 1-D tensor of size floor(end - start)/ + 1 with values from start to end with step step.
    // Step is the gap between two values in the tensor.
    int size = ::floorf((end-start)/step) + 1.0f;
    return raw_range(start, step, size, dev);
}

Tensor* Tensor::linspace(float start, float end, int steps, int dev){
    float step = (end-start)/((float)steps-1.0f + 10e-8f);
    return Tensor::range(start, end, step, dev);
}

Tensor* Tensor::logspace(float start, float end, int steps, float base, int dev){
    float step = (end-start)/((float)steps-1.0f + 10e-8f);
    auto t = Tensor::range(start, end, step, dev);
    t->powb_(base);
    return t;
}

Tensor* Tensor::geomspace(float start, float end, int steps, int dev){
  return Tensor::logspace(::log10f(start), ::log10f(end), steps, 10.0f, dev);
}


Tensor* Tensor::eye(int rows, int offset, int dev){
    auto t = new Tensor(vector<int>{rows, rows}, dev);
    if (t->isCPU()) {
        cpu_eye(t, offset);
    }
#ifdef cGPU
    else if (t->isGPU())
      {
        gpu_eye(t, offset);
      }
#endif
    return t;

}

Tensor* Tensor::identity(int rows, int dev){
    return Tensor::eye(rows, 0, dev);
}


Tensor* Tensor::randu(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_rand_uniform_(1.0f);
    return t;
}

Tensor* Tensor::randn(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_rand_normal_(0.0f, 1.0f, false);
    return t;
}


void Tensor::diag_(int k){
    Tensor::diag(this, this, k);
}

Tensor* Tensor::diag(int k){
    Tensor *t = Tensor::empty_like(this);
    Tensor::diag(this, t, k);
    return t;
}

void Tensor::diag(Tensor* A, Tensor* B, int k){
    checkCompatibility(A, B, "Tensor::diag");

    if(!Tensor::isSquared(A) || A->ndim != 2){  // isSquares is for n dimensions, and here we need just two
        msg("The matrix must be square", "Tensor::diag");
    }

    if (A->isCPU() && B->isCPU()) {
        cpu_diag(A, B, k);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_diag(A, B, k);
    }
#endif
}


