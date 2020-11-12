/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"


void cpu_cent(Tensor *A, Tensor *B, Tensor *C){
  _profile(_CPU_CENT, 0);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    C->ptr[i] = 0;
    if (A->ptr[i] != 0.0) C->ptr[i] -= A->ptr[i] * std::log(B->ptr[i]+0.00001);
    if (A->ptr[i] != 1.0) C->ptr[i] -= (1.0 - A->ptr[i]) * std::log(1.0 - B->ptr[i]+0.00001);
  }
    _profile(_CPU_CENT, 1);
}

float cpu_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred){
    float sum = 0.0f;
    float eps = 10e-8;

    #pragma omp parallel for reduction(+:sum)
    for (int bi = 0; bi<y_true->shape[0]; bi++) {  // Batches
        unsigned int step_i = bi * y_true->stride[0];

        // Compute cross-entropy
        float bi_sum = 0.0f;
        for (unsigned int i = 0; i<y_true->shape[1]; i++) {
            bi_sum += y_true->ptr[step_i + i] * ::logf(y_pred->ptr[step_i + i]+eps);
        }
        sum += bi_sum;
    }

    // Compute mean
    float mean_ce = -sum/(float)y_true->shape[0];
    return mean_ce;
}

void cpu_d_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta){
    float eps = 10e-8;

    #pragma omp parallel for
    for (int i = 0; i<y_true->size; i++) {
        delta->ptr[i] = -y_true->ptr[i] * (1.0f/ (y_pred->ptr[i]+eps) );
    }
}

float cpu_binary_cross_entropy(Tensor* y_true, Tensor* y_pred){
    float sum = 0.0f;
    float eps = 10e-8;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < y_true->size; i++) {
        sum += y_true->ptr[i] * ::logf(y_pred->ptr[i]+eps) + (1.0-y_true->ptr[i]) * ::logf(1.0f-y_pred->ptr[i]+eps);
    }

    // Compute mean
    float mean_ce = -sum/(float)y_true->shape[0];
    return mean_ce;
}

void cpu_d_binary_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta){
    float eps = 10e-8;

    #pragma omp parallel for
    for (int i = 0; i<y_true->size; i++) {
        delta->ptr[i] = -( y_true->ptr[i] * 1.0f/(y_pred->ptr[i]+eps) + (1.0-y_true->ptr[i]) * 1.0f/(1.0f-y_pred->ptr[i]+eps) * -1.0f );
    }
}
