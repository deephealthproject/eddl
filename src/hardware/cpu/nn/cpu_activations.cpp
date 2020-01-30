/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_nn.h"

void cpu_relu(Tensor *A, Tensor *B){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = 0.0;
  }
}

void cpu_d_relu(Tensor *D, Tensor *I, Tensor *PD){
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > 0.0) PD->ptr[i] = D->ptr[i];
    else PD->ptr[i] = 0.0;
  }
}

void cpu_lrelu(Tensor *A, Tensor *B,float param){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = param*A->ptr[i];;
  }
}

void cpu_d_lrelu(Tensor *D, Tensor *I, Tensor *PD,float param){
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > 0.0) PD->ptr[i] = D->ptr[i];
    else PD->ptr[i] = param*D->ptr[i];
  }
}

void cpu_elu(Tensor *A, Tensor *B, float param){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
    else B->ptr[i] = param * (::expf(A->ptr[i]) - 1.0);
  }
}

void cpu_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    if (I->ptr[i] > 0.0) PD->ptr[i] = D->ptr[i];
    else PD->ptr[i] = D->ptr[i] * (param * ::expf(I->ptr[i]));
  }
}

void cpu_linear(Tensor *A, Tensor *B, float param){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    B->ptr[i] = param * A->ptr[i];
  }
}

void cpu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++) {
    PD->ptr[i] = D->ptr[i] * param;
  }
}

void cpu_sigmoid(Tensor *A, Tensor *B){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    B->ptr[i] = 1/(1+std::exp(-A->ptr[i]));
}

void cpu_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] = D->ptr[i]*((1-I->ptr[i])*I->ptr[i]);
}

void cpu_tanh(Tensor *A, Tensor *B){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    float p=std::exp(A->ptr[i]);
    float n=std::exp(-A->ptr[i]);
    B->ptr[i] = (p-n)/(p+n);
  }
}

void cpu_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] = D->ptr[i]*(1-(I->ptr[i]*I->ptr[i]));
}



void cpu_softmax(Tensor *A, Tensor *B) {
  float max, sum;

  //#pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    max = (*A->ptr2).col(i).maxCoeff();
    for (int j = 0; j < A->shape[1]; j++)
    (*B->ptr2)(j, i) = std::exp((*A->ptr2)(j, i) - max);

    sum = (*B->ptr2).col(i).sum();
    for (int j = 0; j < B->shape[1]; j++)
    (*B->ptr2)(j, i) = (*B->ptr2)(j, i) / sum;
  }
}

void cpu_d_softmax(Tensor *D, Tensor *I, Tensor *PD) {
  PD->tsem->lock();

  #pragma omp parallel for
  for (int i = 0; i < D->size; i++)
    PD->ptr[i] += D->ptr[i] * (I->ptr[i] * (1.0 - I->ptr[i]));

  PD->tsem->unlock();
}
