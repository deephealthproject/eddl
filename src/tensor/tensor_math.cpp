
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
#include <cmath>
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
int Tensor::numel(){
    return this->size;
}

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
void Tensor::add(float v) {
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
void Tensor::sub(float v) { add(-v); }



///////////////////////////////////////////
void Tensor::log() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = std::log(ptr[i]);
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
void Tensor::log2() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = std::log2(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_logn(this, 2.0f);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::log10() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = std::log10(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_logn(this, 10.0f);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::logn(float n) {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = std::log10(ptr[i])/std::log10(n);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_logn(this, n);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


///////////////////////////////////////////
void Tensor::abs() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = std::fabs(ptr[i]);
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
void Tensor::exp() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = std::exp(ptr[i]);
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
void Tensor::sqrt() {
    if (isCPU()) {

        for (int i = 0; i < size; ++i) ptr[i] = std::sqrt(ptr[i]);
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
void Tensor::sqr() {
    // pow(x, 2) == x*x  To know more, read comments in pow's function
    // speed: 0.000497s
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

void Tensor::pow(float exp) {
    // To compute the power, std uses real floating-point number with the formurla: e^(y*log(x))
    // Quite inefficient (x100 slower) in g++ except for pow(x, 2) which is inlined as x*x
    // speed: 0.057887s
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = std::pow(ptr[i], exp);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_pow(this, exp);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}



///////////////////////////////////////
float Tensor::sum() {

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
float Tensor::sum_abs() {

    if (isCPU()) {
        float sum = 0.0;

        for (int i = 0; i < size; ++i) sum += std::fabs(ptr[i]);

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
