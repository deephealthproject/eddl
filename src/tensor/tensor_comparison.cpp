/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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

void checkCompatibility(Tensor *A, Tensor *B, const string &title){
    if (A->device != B->device) {
        msg("Tensors in different devices", title);
    }

    if (!Tensor::eqsize(A, B)){
        msg("Tensors with different shape", title);
    }
}


void checkCompatibility(Tensor *A, Tensor *B, Tensor *C, const string &title){
    checkCompatibility(A, B, title);
    checkCompatibility(A, C, title);
}


bool Tensor::allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    checkCompatibility(A, B, "Tensor::allclose");

    if (A->isCPU()) {
        return cpu_allclose(A, B, rtol, atol, equal_nan);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            return gpu_allclose(A, B, rtol, atol, equal_nan);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
    return 0;
}


void Tensor::isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    checkCompatibility(A, B, C, "Tensor::isclose");

    if (A->isCPU()) {
        cpu_isclose(A, B, C, rtol, atol, equal_nan);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_isclose(A, B, C, rtol, atol, equal_nan);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}


void Tensor::greater(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::greater");

    if (A->isCPU()) {
        cpu_greater(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_greater(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}

void Tensor::greater_equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::greater_equal");

    if (A->isCPU()) {
        cpu_greater_equal(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
           gpu_greater_equal(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}

void Tensor::less(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::less");

    if (A->isCPU()) {
        cpu_less(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_less(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}

void Tensor::less_equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::less_equal");

    if (A->isCPU()) {
        cpu_less_equal(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_less_equal(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}

void Tensor::equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::equal");

    if (A->isCPU()) {
        cpu_equal(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_equal(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}

void Tensor::not_equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::not_equal");

    if (A->isCPU()) {
        cpu_not_equal(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_not_equal(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::greater");
        }
#endif
}



int Tensor::eqsize(Tensor *A, Tensor *B) {
    if (A->ndim != B->ndim) return 0;

    for (int i = 0; i < A->ndim; i++)
        if (A->shape[i] != B->shape[i]) return 0;

    return 1;
}

int Tensor::equal2(Tensor *A, Tensor *B, float epsilon) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::equal");

    if (!eqsize(A,B)) return 0;

    if (A->isCPU()) {
        return cpu_equal2(A, B, epsilon);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            msg("Equal only for CPU Tensors", "Tensor::equal");
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::equal");
        }
#endif

    return 1;
}
