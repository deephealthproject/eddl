/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_hw.h"

// CPU: Math (in-place) ********************************************

void cpu_abs_(Tensor *A) {
    _profile(_CPU_ABS_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::fabs(A->ptr[i]);
}

void cpu_acos_(Tensor *A){
    _profile(_CPU_ACOS_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::acosf(A->ptr[i]);
}

void cpu_add_(Tensor *A, float v) {
    _profile(_CPU_ADD_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] += v;
}


void cpu_asin_(Tensor *A){
    _profile(_CPU_ASIN_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::asinf(A->ptr[i]);
}

void cpu_atan_(Tensor *A){
    _profile(_CPU_ATAN_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::atanf(A->ptr[i]);
}

void cpu_ceil_(Tensor *A){
    _profile(_CPU_CEIL_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::ceilf(A->ptr[i]);
}

void cpu_clamp_(Tensor *A, float min, float max){
    _profile(_CPU_CLAMP_);
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
    _profile(_CPU_COS_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::cosf(A->ptr[i]);
}

void cpu_cosh_(Tensor *A){
    _profile(_CPU_COSH_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::coshf(A->ptr[i]);
}

void cpu_exp_(Tensor *A) {
    _profile(_CPU_EXP_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::expf(A->ptr[i]);
}

void cpu_floor_(Tensor *A){
    _profile(_CPU_FLOOR_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::floorf(A->ptr[i]);
}

void cpu_inv_(Tensor *A, float v){
    _profile(_CPU_INV_);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i) A->ptr[i] = v/A->ptr[i];
}

void cpu_log_(Tensor *A) {
    _profile(_CPU_LOG_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::logf(A->ptr[i]);
}

void cpu_log2_(Tensor *A) {
    _profile(_CPU_LOG2_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::log2f(A->ptr[i]);
}

void cpu_log10_(Tensor *A) {
    _profile(_CPU_LOG10_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::log10f(A->ptr[i]);
}

void cpu_logn_(Tensor *A, float n) {
    _profile(_CPU_LOGN_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::logf(A->ptr[i])/::logf(n);
}


void cpu_mod_(Tensor *A, float v){
    _profile(_CPU_MOD_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::fmod(A->ptr[i], v);
}

void cpu_mult_(Tensor *A, float v) {
    _profile(_CPU_MULT_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] *= v;
}

void cpu_normalize_(Tensor *A, float min, float max){
    _profile(_CPU_NORMALIZE_);
  // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
  // (max2-min2)/(max1-min1) * (x-min1) + min2
  float max_ori = A->max();
  float min_ori = A->min();
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = (max-min)/(max_ori-min_ori) * (A->ptr[i]-min_ori) + min;
}

void cpu_pow_(Tensor *A, float exp) {
  _profile(_CPU_POW_);
  // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
  // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
  // speed: 0.057887s
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::powf(A->ptr[i], exp);
}

void cpu_powb_(Tensor *A, float base) {
    _profile(_CPU_POWB_);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i) A->ptr[i] = ::powf(base, A->ptr[i]);
}

void cpu_reciprocal_(Tensor *A) {
    _profile(_CPU_RECIPROCAL_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = 1.0f/A->ptr[i];
}

void cpu_remainder_(Tensor *A, float v) {
    _profile(_CPU_REMAINDER_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = (int)(A->ptr[i]/v);
}

void cpu_round_(Tensor *A){
    _profile(_CPU_ROUND_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::roundf(A->ptr[i]);
}

void cpu_rsqrt_(Tensor *A){
    _profile(_CPU_RSQRT_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = 1.0f/::sqrtf(A->ptr[i]);
}

void cpu_sigmoid_(Tensor *A){
    _profile(_CPU_SIGMOID_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::expf(A->ptr[i])/(::expf(A->ptr[i])+1.0f);
}

void cpu_sign_(Tensor *A){
    _profile(_CPU_SIGN_);
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
    _profile(_CPU_SIN_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::sinf(A->ptr[i]);
}

void cpu_sinh_(Tensor *A){
    _profile(_CPU_SINH_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::sinhf(A->ptr[i]);
}

void cpu_sqr_(Tensor *A) {
    _profile(_CPU_SQR_);
  // pow(x, 2) == x*x  To know more, read comments in pow_'s function
  // speed: 0.000497s
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] *= A->ptr[i];
}

void cpu_sqrt_(Tensor *A) {
    _profile(_CPU_SQRT_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::sqrtf(A->ptr[i]);
}

void cpu_tan_(Tensor *A){
    _profile(_CPU_TAN_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::tanf(A->ptr[i]);
}

void cpu_tanh_(Tensor *A){
  _profile(_CPU_TANH_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::tanhf(A->ptr[i]);
}

void cpu_trunc_(Tensor *A){
  _profile(_CPU_TRUNC_);
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) A->ptr[i] = ::truncf(A->ptr[i]);
}



// CPU: Math (static) ***************************

void cpu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
  _profile(_CPU_ADD);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
    else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
}


void cpu_inc(Tensor *A, Tensor *B) {
  _profile(_CPU_INC);
  B->tsem->lock();

  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    B->ptr[i] += A->ptr[i];

  B->tsem->unlock();
}

void cpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
  _profile(_CPU_MULT2D);
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
  _profile(_CPU_EL_DIV);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
    if (incC) C->ptr[i] += A->ptr[i] / B->ptr[i];
    else C->ptr[i] = A->ptr[i] / B->ptr[i];
}


void cpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile(_CPU_EL_MULT);
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
  if (incC) C->ptr[i] += A->ptr[i] * B->ptr[i];
  else C->ptr[i] = A->ptr[i] * B->ptr[i];
}


void cpu_sign2(Tensor *A, Tensor *B){
  _profile(_CPU_SIGN2);
  // TODO: Remove
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++)
  if (A->ptr[i] < 0) B->ptr[i] = -1.0;
  else B->ptr[i] = 1.0;
}

void cpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile(_CPU_SUM2D_ROWWISE);
  #pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    int p=i*A->shape[1];
    for (int j = 0; j < A->shape[1]; j++, p++)
      C->ptr[p] = A->ptr[p] + B->ptr[j];
  }
}

void cpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile(_CPU_SUM2D_COLWISE);

  #pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    int p=i*A->shape[1];
    for (int j = 0; j < A->shape[1]; j++, p++)
    C->ptr[p] = A->ptr[p] + B->ptr[i];
  }
}

// CPU: Should be reductions ***************************

float cpu_max(Tensor *A){
  _profile(_CPU_MAX);
  float max = MIN_FLOAT;
  // todo: #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) {
    if (A->ptr[i] > max) { max = A->ptr[i]; }
  }
  return max;
}

float cpu_min(Tensor *A){
  _profile(_CPU_MIN);
  float min = MAX_FLOAT;
  // todo: #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) {
    if (A->ptr[i] < min) { min = A->ptr[i]; }
  }
  return min;
}

float cpu_sum(Tensor *A) {
  _profile(_CPU_SUM);
  float sum = 0.0;
  for (int i = 0; i < A->size; ++i) sum += A->ptr[i];
  return sum;
}

float cpu_sum_abs(Tensor *A) {
  _profile(_CPU_SUM_ABS);
  float sum = 0.0;
  for (int i = 0; i < A->size; ++i) sum += ::fabs(A->ptr[i]);
  return sum;
}
