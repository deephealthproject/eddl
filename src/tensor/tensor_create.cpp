/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "../hardware/gpu/gpu_tensor.h"
#include "../hardware/gpu/gpu_hw.h"
#include "../hardware/gpu/nn/gpu_nn.h"
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
#ifdef cFPGA
    else {

    }
#endif
    return t;
}


// ************************************************
// Creation ops ***********************************
// ************************************************

Tensor* Tensor::zeros(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_(0.0f);
    return t;
}

Tensor* Tensor::ones(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_(1.0f);
    return t;
}

Tensor* Tensor::full(const vector<int> &shape, float value, int dev){
    auto t = new Tensor(shape, dev);
    t->fill_(value);
    return t;
}

Tensor* Tensor::arange(float start, float end, float step, int dev){
    // [1, 100)
    // Returns a 1-D tensor of size ceil(end - start) with values from start to end with step step.
    // Step is the gap between two values in the tensor.
    int size = ::ceilf((end-start)/step);
    return raw_range(start, step, size, dev);
}

Tensor* Tensor::range(float start, float end, float step, int dev){
    // [1, 100]
    // Returns a 1-D tensor of size floor(end - start)/ + 1 with values from start to end with step step.
    // Step is the gap between two values in the tensor.
    int size = ::floorf((end-start)/step) + 1;
    return raw_range(start, step, size, dev);
}

Tensor* Tensor::linspace(float start, float end, int steps, int dev){
    float step = (end-start)/((float)steps-1);
    return Tensor::range(start, end, step, dev);
}

Tensor* Tensor::logspace(float start, float end, int steps, float base, int dev){
    float step = (end-start)/((float)steps-1);
    auto t = Tensor::range(start, end, step, dev);
    t->powb_(base);
    return t;
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
#ifdef cFPGA
    else {

    }
#endif
    return t;

}

Tensor* Tensor::identity(int rows, int dev){
    return Tensor::eye(rows, 0, dev);
}


Tensor* Tensor::diag(Tensor* A, int k, int dev){
    msg("Not implemented", "Tensor::diag");

    if(!Tensor::isSquared(A)){
        msg("The matrix must be square", "Tensor::diag");
    }
    auto new_t = new Tensor({A->shape[0]}, dev);
//
//    if (new_t->isCPU()) {
//        cpu_diag(new_t, k);
//    }
//#ifdef cGPU
//    else if (t->isGPU())
//      {
//        gpu_diag(new_t, k);
//      }
//#endif
//#ifdef cFPGA
//    else {
//
//    }
//#endif

    return new_t;
}

Tensor* Tensor::randn(const vector<int> &shape, int dev){
    auto t = new Tensor(shape, dev);
    t->rand_normal(0.0f, 1.0f, false);
    return t;
}


