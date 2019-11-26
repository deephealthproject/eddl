/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "cpu_hw.h"

// CPU: Math (in-place) ********************************************

void cpu_abs_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::fabs(A->ptr[i]);
}

void cpu_acos_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::acosf(A->ptr[i]);
}

void cpu_add_(Tensor *A, float v) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] += v;
}


void cpu_asin_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::asinf(A->ptr[i]);
}

void cpu_atan_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::atanf(A->ptr[i]);
}

void cpu_ceil_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::ceilf(A->ptr[i]);
}

void cpu_clamp_(Tensor *A, float min, float max){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i){
    if (A->ptr[i] < min){
      A->ptr[i] = min;
    } else if(A->ptr[i] > max){
      A->ptr[i] = max;
    }
  }
}


void cpu_cos_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::cosf(A->ptr[i]);
}

void cpu_cosh_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::coshf(A->ptr[i]);
}

void cpu_exp_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::expf(A->ptr[i]);
}

void cpu_floor_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::floorf(A->ptr[i]);
}

void cpu_inv_(Tensor *A){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) A->ptr[i] = 1/A->ptr[i];
}

void cpu_log_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::logf(A->ptr[i]);
}

void cpu_log2_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::log2f(A->ptr[i]);
}

void cpu_log10_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::log10f(A->ptr[i]);
}

void cpu_logn_(Tensor *A, float n) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::logf(A->ptr[i])/::logf(n);
}


void cpu_mod_(Tensor *A, float v){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::fmod(A->ptr[i], v);
}

void cpu_mult_(Tensor *A, float v) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] *= v;
}

void cpu_normalize_(Tensor *A, float min, float max){
  // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
  // (max2-min2)/(max1-min1) * (x-min1) + min2
  float max_ori = A->max();
  float min_ori = A->min();
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = (max-min)/(max_ori-min_ori) * (A->ptr[i]-min_ori) + min;
}

void cpu_pow_(Tensor *A, float exp) {
  // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
  // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
  // speed: 0.057887s
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::powf(A->ptr[i], exp);
}

void cpu_powb_(Tensor *A, float base) {
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i) A->ptr[i] = ::powf(base, A->ptr[i]);
}

void cpu_reciprocal_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = 1.0f/A->ptr[i];
}

void cpu_remainder_(Tensor *A, float v) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = (int)(A->ptr[i]/v);
}

void cpu_round_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::roundf(A->ptr[i]);
}

void cpu_rsqrt_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = 1.0f/::sqrtf(A->ptr[i]);
}

void cpu_sigmoid_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::expf(A->ptr[i])/(::expf(A->ptr[i])+1.0f);
}

void cpu_sign_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) {
    if(A->ptr[i] > 0.0f){
      A->ptr[i] = 1.0f;
    }else if(A->ptr[i] < 0.0f){
      A->ptr[i] = -1.0f;
    }else{
      A->ptr[i] = 0.0f;
    }
  };
}


void cpu_sin_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::sinf(A->ptr[i]);
}

void cpu_sinh_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::sinhf(A->ptr[i]);
}

void cpu_sqr_(Tensor *A) {
  // pow(x, 2) == x*x  To know more, read comments in pow_'s function
  // speed: 0.000497s
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] *= A->ptr[i];
}

void cpu_sqrt_(Tensor *A) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::sqrtf(A->ptr[i]);
}

void cpu_tan_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::tanf(A->ptr[i]);
}

void cpu_tanh_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::tanhf(A->ptr[i]);
}

void cpu_trunc_(Tensor *A){
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::truncf(A->ptr[i]);
}



// CPU: Math (static) ***************************

void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
    else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
}


void cpu_inc(Tensor *A, Tensor *B) {
  B->tsem->lock();

  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    B->ptr[i] += A->ptr[i];

  B->tsem->unlock();
}

void cpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
  if (!tB) {
    if (!tA) {
      if (!incC) *(C->ptr2) = *(B->ptr2) * (*(A->ptr2));
      else *(C->ptr2) += *(B->ptr2) * (*(A->ptr2));
    } else {
      if (!incC) *(C->ptr2) = *(B->ptr2) * ((*(A->ptr2)).transpose());
      else *(C->ptr2) += *(B->ptr2) * ((*(A->ptr2)).transpose());
    }
  } else {
    if (!tA) {
      if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * (*(A->ptr2));
      else *(C->ptr2) += (*(B->ptr2)).transpose() * (*(A->ptr2));
    } else {
      if (!incC) *(C->ptr2) = (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
      else *(C->ptr2) += (*(B->ptr2)).transpose() * ((*(A->ptr2)).transpose());
    }
  }
}

void cpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    if (incC) C->ptr[i] += A->ptr[i] / B->ptr[i];
    else C->ptr[i] = A->ptr[i] / B->ptr[i];
}


void cpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
  if (incC) C->ptr[i] += A->ptr[i] * B->ptr[i];
  else C->ptr[i] = A->ptr[i] * B->ptr[i];
}


void cpu_sign2(Tensor *A, Tensor *B){
  // TODO: Remove
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
  if (A->ptr[i] < 0) B->ptr[i] = -1.0;
  else B->ptr[i] = 1.0;
}

void cpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  #pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    int p=i*A->shape[1];
    for (int j = 0; j < A->shape[1]; j++, p++)
      C->ptr[p] = A->ptr[p] + B->ptr[j];
  }
}

void cpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {

  #pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    int p=i*A->shape[1];
    for (int j = 0; j < A->shape[1]; j++, p++)
    C->ptr[p] = A->ptr[p] + B->ptr[i];
  }
}

// CPU: Should be reductions ***************************

float cpu_max(Tensor *A){
  float max = MIN_FLOAT;
  // todo: #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) {
    if (A->ptr[i] > max) { max = A->ptr[i]; }
  }
  return max;
}

float cpu_min(Tensor *A){
  float min = MAX_FLOAT;
  // todo: #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) {
    if (A->ptr[i] < min) { min = A->ptr[i]; }
  }
  return min;
}

float cpu_sum(Tensor *A) {
  float sum = 0.0;
  for (int i = 0; i < A->size; ++i) sum += A->ptr[i];
  return sum;
}

float cpu_sum_abs(Tensor *A) {
  float sum = 0.0;
  for (int i = 0; i < A->size; ++i) sum += ::fabs(A->ptr[i]);
  return sum;
}
