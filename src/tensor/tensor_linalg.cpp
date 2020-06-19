/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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


float Tensor::trace(int k){
    return Tensor::trace(this, k);
}

float Tensor::trace(Tensor *A, int k){
    Tensor *t = A->diag(k);  // Generate diagonal

    float sum_diag = t->sum();
    delete t;

    return sum_diag;
}

float Tensor::norm(string ord){
    return Tensor::norm(this, ord);
}

float Tensor::norm(Tensor *A, string ord){
    if (A->isCPU()) {
        return cpu_norm(A, ord);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        return gpu_norm(A, ord);
    }
#endif
#ifdef cFPGA
    else {

    }
#endif
    return 0.0f;
}


Tensor* Tensor::norm(vector<int> axis, bool keepdims, string ord){
    // Build descriptor
    auto rd = new ReduceDescriptor2(axis, keepdims, this->device);
    rd->build(this->shape);

    // Create output tensor
    Tensor *t = Tensor::empty(rd->oshape, this->device);
    Tensor::norm(this, t, rd, ord);

    delete rd;
    return t;
}

void Tensor::norm(Tensor* A, Tensor *B, ReduceDescriptor2 *rd, string ord){
    if (A->isCPU() && B->isCPU()) {
        cpu_norm(A, B, rd, ord);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU())
    {
        gpu_norm(A, B, rd, ord);
    }
#endif
#ifdef cFPGA
    else {

    }
#endif
}