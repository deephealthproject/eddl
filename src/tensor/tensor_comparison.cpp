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
#include "eddl/profiling.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

using namespace std;

PROFILING_ENABLE_EXTERN(all);
PROFILING_ENABLE_EXTERN(any);
PROFILING_ENABLE_EXTERN(isfinite);
PROFILING_ENABLE_EXTERN(isinf);
PROFILING_ENABLE_EXTERN(isnan);
PROFILING_ENABLE_EXTERN(isneginf);
PROFILING_ENABLE_EXTERN(isposinf);
PROFILING_ENABLE_EXTERN(logical_and);
PROFILING_ENABLE_EXTERN(logical_or);
PROFILING_ENABLE_EXTERN(logical_not);
PROFILING_ENABLE_EXTERN(logical_xor);
PROFILING_ENABLE_EXTERN(allclose);
PROFILING_ENABLE_EXTERN(isclose);
PROFILING_ENABLE_EXTERN(greater);
PROFILING_ENABLE_EXTERN(greater_equal);
PROFILING_ENABLE_EXTERN(less);
PROFILING_ENABLE_EXTERN(less_equal);
PROFILING_ENABLE_EXTERN(equal);
PROFILING_ENABLE_EXTERN(not_equal);
PROFILING_ENABLE_EXTERN(equivalent);


bool Tensor::all(){
    return Tensor::all(this);
}
bool Tensor::any(){
    return Tensor::any(this);
}


bool Tensor::all(Tensor *A){

    PROFILING_HEADER(all);

    bool res = false;

    if (A->isCPU()) {
        res = cpu_all(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        res = gpu_all(A);
    }
#endif
#ifdef cFPGA
    else {
        res = fpga_all(A);
    }
#endif

    PROFILING_FOOTER(all);

    return res;
}

bool Tensor::any(Tensor *A){

    PROFILING_HEADER(any);

    bool res = false;

    if (A->isCPU()) {
        res = cpu_any(A);
    }
#ifdef cGPU
    else if (A->isGPU())
    {
        res = gpu_any(A);
    }
#endif
#ifdef cFPGA
    else {
        res = fpga_any(A);
    }
#endif

   PROFILING_FOOTER(any);

   return res;
}

// Logic funcions: Logical ops
void Tensor::isfinite(Tensor *A, Tensor* B){
    checkCompatibility(A, B, "Tensor::isfinite");

    PROFILING_HEADER(isfinite);

    if (A->isCPU()) {
        cpu_isfinite(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
          gpu_isfinite(A, B);
          }
#endif
#ifdef cFPGA
    else {
         fpga_isfinite(A, B);
        }
#endif
    PROFILING_FOOTER(isfinite);
}

void Tensor::isinf(Tensor *A, Tensor* B){
    checkCompatibility(A, B, "Tensor::isinf");

    PROFILING_HEADER(isinf);

    if (A->isCPU()) {
        cpu_isinf(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
          gpu_isinf(A, B);
          }
#endif
#ifdef cFPGA
    else {
          fpga_isinf(A, B);
        }
#endif
    PROFILING_FOOTER(isinf);
}

void Tensor::isnan(Tensor *A, Tensor* B){
    checkCompatibility(A, B, "Tensor::isnan");

    PROFILING_HEADER(isnan);

    if (A->isCPU()) {
        cpu_isnan(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
          gpu_isnan(A, B);
          }
#endif
#ifdef cFPGA
    else {
          fpga_isnan(A, B);
        }
#endif
    PROFILING_FOOTER(isnan);
}

void Tensor::isneginf(Tensor *A, Tensor* B){
    checkCompatibility(A, B, "Tensor::isneginf");

    PROFILING_HEADER(isneginf);

    if (A->isCPU()) {
        cpu_isneginf(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
          gpu_isneginf(A, B);
          }
#endif
#ifdef cFPGA
    else {
         fpga_isneginf(A, B);
        }
#endif
    PROFILING_FOOTER(isneginf);
}

void Tensor::isposinf(Tensor *A, Tensor* B){
    checkCompatibility(A, B, "Tensor::isposinf");

    PROFILING_HEADER(isposinf);

    if (A->isCPU()) {
        cpu_isposinf(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
          gpu_isposinf(A, B);
          }
#endif
#ifdef cFPGA
    else {
         fpga_isposinf(A, B);
        }
#endif
    PROFILING_FOOTER(isposinf);
}


// Logic funcions: Logical ops

void Tensor::logical_and(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::logical_and");

    PROFILING_HEADER(logical_and);

    if (A->isCPU()) {
        cpu_logical_and(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_logical_and(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          fpga_logical_and(A, B, C);
        }
#endif
    PROFILING_FOOTER(logical_and);
}

void Tensor::logical_or(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::logical_or");

    PROFILING_HEADER(logical_or);

    if (A->isCPU()) {
        cpu_logical_or(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_logical_or(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          fpga_logical_or(A, B, C);
        }
#endif
    PROFILING_FOOTER(logical_or);
}

void Tensor::logical_not(Tensor *A, Tensor *B){
    checkCompatibility(A, B, "Tensor::logical_not");

    PROFILING_HEADER(logical_not);

    if (A->isCPU()) {
        cpu_logical_not(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_logical_not(A, B);
          }
#endif
#ifdef cFPGA
    else {
          fpga_logical_not(A, B);
        }
#endif
    PROFILING_FOOTER(logical_not);
}

void Tensor::logical_xor(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::logical_xor");

    PROFILING_HEADER(logical_xor);

    if (A->isCPU()) {
        cpu_logical_xor(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_logical_xor(A, B, C);
          }
#endif
#ifdef cFPGA
    else {
          fpga_logical_xor(A, B, C);
        }
#endif
    PROFILING_FOOTER(logical_xor);
}



bool Tensor::allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    checkCompatibility(A, B, "Tensor::allclose");

    PROFILING_HEADER(allclose);

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
          return fpga_allclose(A, B, rtol, atol, equal_nan);
        }
#endif
    PROFILING_FOOTER(allclose);

    return 0;
}


void Tensor::isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    checkCompatibility(A, B, C, "Tensor::isclose");

    PROFILING_HEADER(isclose);

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
          fpga_isclose(A, B, C, rtol, atol, equal_nan);
        }
#endif
    PROFILING_FOOTER(isclose);
}

void Tensor::greater_(float v){
    Tensor::greater(this, this, v);
}

Tensor* Tensor::greater(float v){
    Tensor *t = this->clone();
    t->greater_(v);
    return t;
}

void Tensor::greater(Tensor *A, Tensor *B, float v){
    checkCompatibility(A, B, "Tensor::greater");

    PROFILING_HEADER(greater);

    if (A->isCPU()) {
        cpu_greater(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_greater(A, B, v);
          }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
          {
            fpga_greater(A, B, v);
          }
#endif

    PROFILING_FOOTER(greater);
}

Tensor* Tensor::greater(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->greater(this, A, t);
    return t;
}

void Tensor::greater(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::greater");

    PROFILING_HEADER(greater);

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
            fpga_greater(A, B, C);
        }
#endif

    PROFILING_FOOTER(greater);
}


void Tensor::greater_equal_(float v){
    Tensor::greater_equal(this, this, v);
}

Tensor* Tensor::greater_equal(float v){
    Tensor *t = this->clone();
    t->greater_equal_(v);
    return t;
}

void Tensor::greater_equal(Tensor *A, Tensor *B, float v){
    checkCompatibility(A, B, "Tensor::greater_equal");

    PROFILING_HEADER(greater_equal);

    if (A->isCPU()) {
        cpu_greater_equal(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_greater_equal(A, B, v);
          }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
          {
            fpga_greater_equal(A, B, v);
          }
#endif
    PROFILING_FOOTER(greater_equal);
}

Tensor* Tensor::greater_equal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->greater_equal(this, A, t);
    return t;
}

void Tensor::greater_equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::greater_equal");

    PROFILING_HEADER(greater_equal);

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
    else if (A->isFPGA())
        {
        fpga_greater_equal(A, B, C);
        }
#endif

    PROFILING_FOOTER(greater_equal);
}


void Tensor::less_(float v){
    Tensor::less(this, this, v);
}

Tensor* Tensor::less(float v){
    Tensor *t = this->clone();
    t->less_(v);
    return t;
}

void Tensor::less(Tensor *A, Tensor *B, float v){
    checkCompatibility(A, B, "Tensor::less");

    PROFILING_HEADER(less);

    if (A->isCPU()) {
        cpu_less(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_less(A, B, v);
          }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
         {
             fpga_less(A, B, v);
        }
#endif
    PROFILING_FOOTER(less);
}

Tensor* Tensor::less(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->less(this, A, t);
    return t;
}

void Tensor::less(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::less");

    PROFILING_HEADER(less);

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
         fpga_less(A, B, C);
        }
#endif

    PROFILING_FOOTER(less);
}


void Tensor::less_equal_(float v){
    Tensor::equal(this, this, v);
}

Tensor* Tensor::less_equal(float v){
    Tensor *t = this->clone();
    t->equal_(v);
    return t;
}

void Tensor::less_equal(Tensor *A, Tensor *B, float v){
    checkCompatibility(A, B, "Tensor::less_equal");

    PROFILING_HEADER(less_equal);

    if (A->isCPU()) {
        cpu_less_equal(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_less_equal(A, B, v);
          }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
          {
            fpga_less_equal(A, B, v);
          }
#endif

    PROFILING_FOOTER(less_equal);
}


Tensor* Tensor::less_equal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->less_equal(this, A, t);
    return t;
}

void Tensor::less_equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::less_equal");

    PROFILING_HEADER(less_equal);

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
          fpga_less_equal(A, B, C);
        }
#endif
    PROFILING_FOOTER(less_equal);
}


void Tensor::equal_(float v){
    Tensor::equal(this, this, v);
}

Tensor* Tensor::equal(float v){
    Tensor *t = this->clone();
    t->equal_(v);
    return t;
}

void Tensor::equal(Tensor *A, Tensor *B, float v){
    checkCompatibility(A, B, "Tensor::equal");

    PROFILING_HEADER(equal);

    if (A->isCPU()) {
        cpu_equal(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_equal(A, B, v);
          }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
          {
            fpga_equal(A, B, v);
          }
#endif

    PROFILING_FOOTER(equal);
}

Tensor* Tensor::equal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->equal(this, A, t);
    return t;
}

void Tensor::equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::equal");

    PROFILING_HEADER(equal);

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
         fpga_equal(A, B, C);
        }
#endif

    PROFILING_FOOTER(equal);
}

void Tensor::not_equal_(float v){
    Tensor::not_equal(this, this, v);
}

Tensor* Tensor::not_equal(float v){
    Tensor *t = this->clone();
    t->not_equal_(v);
    return t;
}

void Tensor::not_equal(Tensor *A, Tensor *B, float v){
    checkCompatibility(A, B, "Tensor::not_equal");

    PROFILING_HEADER(not_equal);

    if (A->isCPU()) {
        cpu_not_equal(A, B, v);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            gpu_not_equal(A, B, v);
          }
#endif
#ifdef cFPGA
    else if (A->isFPGA())
          {
            fpga_not_equal(A, B, v);
          }
#endif

    PROFILING_FOOTER(not_equal);
}

Tensor* Tensor::not_equal(Tensor *A){
    Tensor *t = Tensor::empty_like(this);
    t->not_equal(this, A, t);
    return t;
}

void Tensor::not_equal(Tensor *A, Tensor *B, Tensor *C){
    checkCompatibility(A, B, C, "Tensor::not_equal");

    PROFILING_HEADER(not_equal);

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
           fpga_not_equal(A, B, C);
        }
#endif

    PROFILING_FOOTER(not_equal);
}

int Tensor::eqsize(Tensor *A, Tensor *B){
    return Tensor::sameShape(A, B);
}


bool Tensor::sameSize(Tensor *A, Tensor *B) {
    return A->size == B->size;
}

bool Tensor::sameDevice(Tensor *A, Tensor *B) {
    return A->device == B->device;
}

int Tensor::sameShape(Tensor *A, Tensor *B) {
    if (A->ndim != B->ndim) return 0;

    for (int i = 0; i < A->ndim; i++){
        if (A->shape[i] != B->shape[i]) return 0;
    }

    return 1;
}

int Tensor::equivalent(Tensor *A, Tensor *B, float atol, float rtol, bool equal_nan) {

    // Equal device
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::equivalent");

    // Equal ndims and shapes
    if (!sameShape(A, B)) return 0;

    PROFILING_HEADER(equivalent);
    
    // Equal data
    if (A->isCPU() && B->isCPU()) {
//        return cpu_allclose(A, B, rtol, atol, equal_nan);
        return cpu_equal2(A, B, atol);  // TODO: Temp!
    }
#ifdef cGPU
    else if (A->isGPU() || B->isGPU())
          {
              return gpu_allclose(A, B, rtol, atol, equal_nan);
          }
#endif
#ifdef cFPGA
    else {
	  printf("Error, please check (FPGA), epsilon does not exist\n");
//          return fpga_equal2(A, B, epsilon);
        }
#endif

    PROFILING_FOOTER(equivalent);

    return 1;
}
