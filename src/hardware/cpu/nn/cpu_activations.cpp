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

void cpu_relu(Tensor *A, Tensor *B){
    _profile(_CPU_RELU, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
        else B->ptr[i] = 0.0;
    }
    _profile(_CPU_RELU, 1);
}

void cpu_d_relu(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_RELU, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        if (I->ptr[i] > 0.0) PD->ptr[i] += D->ptr[i];
        else PD->ptr[i] += 0.0;
    }
    _profile(_CPU_D_RELU, 1);
}

void cpu_thresholded_relu(Tensor *A, Tensor *B,float param){
    _profile(_CPU_THRESHOLDED_RELU, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        if (A->ptr[i] > param) B->ptr[i] = A->ptr[i];
        else B->ptr[i] = 0.0;
    }
    _profile(_CPU_THRESHOLDED_RELU, 1);
}

void cpu_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
    _profile(_CPU_D_THRESHOLDED_RELU, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        if (I->ptr[i] > param) PD->ptr[i] += D->ptr[i];
        else PD->ptr[i] += 0.0;
    }
    _profile(_CPU_D_THRESHOLDED_RELU, 1);
}

void cpu_leaky_relu(Tensor *A, Tensor *B,float param){
    _profile(_CPU_LEAKY_RELU, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
        else B->ptr[i] = param*A->ptr[i];;
    }
    _profile(_CPU_LEAKY_RELU, 1);
}

void cpu_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
    _profile(_CPU_D_LEAKY_RELU, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        if (I->ptr[i] > 0.0) PD->ptr[i] += D->ptr[i];
        else PD->ptr[i] += param*D->ptr[i];
    }
    _profile(_CPU_D_LEAKY_RELU, 1);
}

void cpu_elu(Tensor *A, Tensor *B, float param){
    _profile(_CPU_ELU, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
        else B->ptr[i] = param * (::expf(A->ptr[i]) - 1.0);
    }
    _profile(_CPU_ELU, 1);
}

void cpu_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
    _profile(_CPU_D_ELU, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        if (I->ptr[i] > 0.0) PD->ptr[i] += D->ptr[i];
        else PD->ptr[i] += D->ptr[i] * (param * ::expf(I->ptr[i]));
    }
    _profile(_CPU_D_ELU, 1);
}

void cpu_softplus(Tensor *A, Tensor *B){
    _profile(_CPU_SOFTPLUS, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        B->ptr[i] = ::logf(1 + ::expf(A->ptr[i]));
    }
    _profile(_CPU_SOFTPLUS, 1);
}

void cpu_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_SOFTPLUS, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        PD->ptr[i] += D->ptr[i] * 1/(1 + ::expf(-I->ptr[i]));
    }
    _profile(_CPU_D_SOFTPLUS, 1);
}

void cpu_softsign(Tensor *A, Tensor *B){
    _profile(_CPU_SOFTSIGN, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        B->ptr[i] = A->ptr[i] / (1 + ::fabs(A->ptr[i]));
    }
    _profile(_CPU_SOFTSIGN, 1);
}

void cpu_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_SOFTSIGN, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        float denom = 1 + ::fabs(I->ptr[i]);
        PD->ptr[i] += D->ptr[i] * 1/(denom*denom);
    }
    _profile(_CPU_D_SOFTSIGN, 1);
}

void cpu_linear(Tensor *A, Tensor *B, float param){
    _profile(_CPU_LINEAR, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        B->ptr[i] = param * A->ptr[i];
    }
    _profile(_CPU_LINEAR, 1);
}

void cpu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
    _profile(_CPU_D_LINEAR, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++) {
        PD->ptr[i] += D->ptr[i] * param;
    }
    _profile(_CPU_D_LINEAR, 1);
}

//void cpu_sigmoid(Tensor *A, Tensor *B){
//  _profile(_CPU_SIGMOID, 0);
//  #pragma omp parallel for
//  for (int i = 0; i < A->size; i++)
//    B->ptr[i] = 1/(1+std::exp(-A->ptr[i]));
//    _profile(_CPU_SIGMOID, 1);
//}

void cpu_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_SIGMOID, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++)
        PD->ptr[i] += D->ptr[i]*((1-I->ptr[i])*I->ptr[i]);
    _profile(_CPU_D_SIGMOID, 1);
}

void cpu_hard_sigmoid(Tensor *A, Tensor *B){
    _profile(_CPU_HARD_SIGMOID, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        if (A->ptr[i] > 2.5) B->ptr[i] = 1.0;
        else if (A->ptr[i] < -2.5) B->ptr[i] = 0.0;
        else B->ptr[i] = (0.2 * A->ptr[i]) + 0.5;
    }
    _profile(_CPU_HARD_SIGMOID, 1);
}

void cpu_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_HARD_SIGMOID, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++)
        if (I->ptr[i] < -2.5 || I->ptr[i] > 2.5) PD->ptr[i] += 0;
        else PD->ptr[i] += D->ptr[i] * 0.2;
    _profile(_CPU_D_HARD_SIGMOID, 1);
}

//void cpu_exp(Tensor *A, Tensor *B){
//  _profile(_CPU_EXP, 0);
//  #pragma omp parallel for
//  for (int i = 0; i < A->size; i++) {
//    B->ptr[i] = std::exp(A->ptr[i]);
//  }
//    _profile(_CPU_EXP, 1);
//}

void cpu_d_exp(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_EXP, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++)
        PD->ptr[i] += D->ptr[i] * I->ptr[i];
    _profile(_CPU_D_EXP, 1);
}

//void cpu_tanh(Tensor *A, Tensor *B){
//  _profile(_CPU_TANH, 0);
//  #pragma omp parallel for
//  for (int i = 0; i < A->size; i++) {
//    float p=std::exp(A->ptr[i]);
//    float n=std::exp(-A->ptr[i]);
//    B->ptr[i] = (p-n)/(p+n);
//  }
//    _profile(_CPU_TANH, 1);
//}

void cpu_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
    _profile(_CPU_D_TANH, 0);
#pragma omp parallel for
    for (int i = 0; i < D->size; i++)
        PD->ptr[i] += D->ptr[i]*(1-(I->ptr[i]*I->ptr[i]));
    _profile(_CPU_D_TANH, 1);
}


void cpu_softmax(Tensor *A, Tensor *B) {
    _profile(_CPU_SOFTMAX, 0);
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
    _profile(_CPU_SOFTMAX, 1);
}

void cpu_d_softmax(Tensor *D, Tensor *I, Tensor *PD) {
    _profile(_CPU_D_SOFTMAX, 0);
    PD->tsem->lock();

#pragma omp parallel for
    for (int i = 0; i < D->size; i++)
        PD->ptr[i] += D->ptr[i] * (I->ptr[i] * (1.0 - I->ptr[i]));

    PD->tsem->unlock();
    _profile(_CPU_D_SOFTMAX, 1);
}


void cpu_full_softmax(Tensor *A, Tensor *B, int axis, bool stable){
    if(axis==1){
        cpu_full_softmax_batched(A, B, stable);
    }else{ msg("Not implemented Error", "cpu_full_softmax"); }
}

void cpu_full_softmax_batched(Tensor *A, Tensor *B, bool stable){
    int n_batches = A->shape[0];
    int n_features = A->shape[1];

//    #pragma omp parallel for
    for(int bi=0; bi<n_batches; bi++){
        // Contiguous data
        int start = bi*n_features;
        int end = start+n_features;  // x < end or x <= end-1

        // Numerical stability (opt.)
        // stable => first value, no stable => 0.0f
        float max_value = 0.0f;
        if(stable){
            for(int j=start; j<end; j++){
                if (A->ptr[j] > max_value) { max_value = A->ptr[j]; }
            }
        }

        // Numerator
        float denominator = 0.0f;
        for(int j=start; j<end; j++){
            float value = ::expf(A->ptr[j] - max_value);
            B->ptr[j] = value;
            denominator += value;
        }

        // Softmax
        for(int j=start; j<end; j++){
            B->ptr[j] /= denominator;
        }
    }
}

void cpu_d_full_softmax(Tensor *D, Tensor *I, Tensor *PD, int axis) {
    if(axis==1){
        cpu_d_full_softmax_batched(D, I, PD);
    }else{ msg("Not implemented Error", "cpu_d_full_softmax"); }
}

void cpu_d_full_softmax_batched(Tensor *D, Tensor *I, Tensor *PD) {
    Tensor* SM = I; // Alias (softmax)

    int n_batches = D->shape[0];
    int n_features = D->shape[1];

//    #pragma omp parallel for
    for(int bi=0; bi<n_batches; bi++){
        // Contiguous data
        int start = bi*n_features;

        // 1) Compute Jacobbian matrix: DS=[ NxN ]  // DjSi
        // 2) Compute delta: D * DS = (1,n)x(n,n)=(1,n)
        // 2.1) Dot product: D0*DS0,0 + D1*DS0,1 + D2*DS0,2 + ...
        for(int i=0; i<n_features; i++){  // Rows
            for(int j=0; j<n_features; j++){  // Cols

                // Derivative
                float DjSi = SM->ptr[i] * ((float)(i==j) - SM->ptr[j]);

                // Dot product: Dj=D*DS[:, j]
                // "i" trick. Technically, PD is (1, n) but I can consider it as (n, 1) without reshaping it
                D->ptr[start+j] += PD->ptr[start+i] * DjSi;
            }
        }
    }
}