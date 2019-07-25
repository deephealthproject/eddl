
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
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

