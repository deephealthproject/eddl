/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_hw.h"

// CPU: Math (in-place) ********************************************

void cpu_abs(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::fabs(A->ptr[i]);
}

void cpu_acos(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::acosf(A->ptr[i]);
}

void cpu_add(Tensor *A, Tensor *B, float v) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] += v;
}


void cpu_asin(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::asinf(A->ptr[i]);
}

void cpu_atan(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::atanf(A->ptr[i]);
}

void cpu_ceil(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::ceilf(A->ptr[i]);
}

void cpu_clamp(Tensor *A, Tensor *B, float min, float max){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if (A->ptr[i] < min){
            B->ptr[i] = min;
        } else if(A->ptr[i] > max){
            B->ptr[i] = max;
        }
    }
}


void cpu_cos(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::cosf(A->ptr[i]);
}

void cpu_cosh(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::coshf(A->ptr[i]);
}

void cpu_exp(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::expf(A->ptr[i]);
}

void cpu_floor(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::floorf(A->ptr[i]);
}

void cpu_inv(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = v/A->ptr[i];
}

void cpu_log(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::logf(A->ptr[i]);
}

void cpu_log2(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::log2f(A->ptr[i]);
}

void cpu_log10(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::log10f(A->ptr[i]);
}

void cpu_logn(Tensor *A, Tensor *B, float n) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::logf(A->ptr[i])/::logf(n);
}


void cpu_mod(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::fmod(A->ptr[i], v);
}

void cpu_mult(Tensor *A, Tensor *B, float v) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] *= v;
}

void cpu_normalize(Tensor *A, Tensor *B, float min, float max){
    // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
    // (max2-min2)/(max1-min1) * (x-min1) + min2
    float max_ori = A->max();
    float min_ori = A->min();
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = (max-min)/(max_ori-min_ori) * (A->ptr[i]-min_ori) + min;
}

void cpu_pow(Tensor *A, Tensor *B, float exp) {
    // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
    // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
    // speed: 0.057887s
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::powf(A->ptr[i], exp);
}

void cpu_powb(Tensor *A, Tensor *B, float base) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::powf(base, A->ptr[i]);
}

void cpu_reciprocal(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = 1.0f/A->ptr[i];
}

void cpu_remainder(Tensor *A, Tensor *B, float v) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = (int)(A->ptr[i]/v);
}

void cpu_round(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::roundf(A->ptr[i]);
}

void cpu_rsqrt(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = 1.0f/::sqrtf(A->ptr[i]);
}

void cpu_sigmoid(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::expf(A->ptr[i])/(::expf(A->ptr[i])+1.0f);
}

void cpu_sign(Tensor *A, Tensor *B, float zero_sign){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) {
        if(A->ptr[i] > 0.0f){
            B->ptr[i] = 1.0f;
        }else if(A->ptr[i] < 0.0f){
            B->ptr[i] = -1.0f;
        }else{
            B->ptr[i] = zero_sign;  // 0.0f recommended
        }
    };
}


void cpu_sin(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::sinf(A->ptr[i]);
}

void cpu_sinh(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::sinhf(A->ptr[i]);
}

void cpu_sqr(Tensor *A, Tensor *B) {
    // pow(x, 2) == x*x  To know more, read comments in pow_'s function
    // speed: 0.000497s
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] *= A->ptr[i];
}

void cpu_sqrt(Tensor *A, Tensor *B) {
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::sqrtf(A->ptr[i]);
}

void cpu_tan(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::tanf(A->ptr[i]);
}

void cpu_tanh(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::tanhf(A->ptr[i]);
}

void cpu_trunc(Tensor *A, Tensor *B){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) B->ptr[i] = ::truncf(A->ptr[i]);
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
    for (int i = 0; i < A->size; i++){
        B->ptr[i] += A->ptr[i];
    }

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
