/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cmath>
#include <limits>
#include <iostream>

#include "eddl/tensor/tensor.h"
#include "eddl/hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_nn.h"
#endif


using namespace std;



void Tensor::abs(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_abs(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_abs(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::acos(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_acos(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_acos(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::asin(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_asin(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_asin(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::atan(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_atan(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_atan(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::ceil(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_ceil(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_ceil(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::clamp(Tensor *A, Tensor *B, float min, float max){
    if (A->isCPU() && B->isCPU()) {
        cpu_clamp(A, B, min, max);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_clamp(A, B, min, max);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::clampmax(Tensor *A, Tensor *B, float max){
    Tensor::clamp(A, B, MIN_FLOAT, max);
}

void Tensor::clampmin(Tensor *A, Tensor *B, float min){
    Tensor::clamp(A, B, min, MAX_FLOAT);
}


void Tensor::cos(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_cos(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_cos(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::cosh(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_cosh(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_cosh(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::inv(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_inv(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_inv(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::exp(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_exp(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_exp(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::floor(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_floor(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_floor(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::log(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_log(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_log(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::log2(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_log2(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_log2(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::log10(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_log10(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_log10(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::logn(Tensor *A, Tensor *B, float n){
    if (A->isCPU() && B->isCPU()) {
        cpu_logn(A, B, n);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_logn(A, B, n);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::mod(Tensor *A, Tensor *B, float v){
    if (A->isCPU() && B->isCPU()) {
        cpu_mod(A, B, v);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_mod(A, B, v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::neg(Tensor *A, Tensor *B){
    Tensor::mult(A, B, -1.0f);
}


void Tensor::normalize(Tensor *A, Tensor *B, float min, float max){
    if (A->isCPU() && B->isCPU()) {
        cpu_normalize(A, B, min, max);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        cpu_normalize(A, B, min, max);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::pow(Tensor *A, Tensor *B, float exp){
    if (A->isCPU() && B->isCPU()) {
        cpu_pow(A, B, exp);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        cpu_pow(A, B, exp);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}



void Tensor::powb(Tensor *A, Tensor *B, float base){
    if (A->isCPU() && B->isCPU()) {
        cpu_powb(A, B, base);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        cpu_powb(A, B, base);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::reciprocal(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_reciprocal(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_reciprocal(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::remainder(Tensor *A, Tensor *B, float v){
    if (A->isCPU() && B->isCPU()) {
        cpu_remainder(A, B, v);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_remainder(A, B, v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::round(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_round(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_round(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::rsqrt(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_rsqrt(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_rsqrt(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::sigmoid(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_sigmoid(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_sigmoid(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::sign(Tensor *A, Tensor *B) {
    ///////////////////////////////////////
    /// Get sign (+-) of all values
    //////////////////////////////////////
    //B->tsem->lock();

    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::sign");
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::sign");

    if (A->isCPU()) {
        cpu_sign2(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        Tensor::copy(A,B);
        gpu_sign_(B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    //B->tsem->unlock();

}


void Tensor::sin(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_sin(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_sin(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::sinh(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_sinh(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_sinh(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::sqr(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_sqr(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_sqr(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::sqrt(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_log10(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_sqrt(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::sub(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_sub(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_sub(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::tan(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_tan(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_tan(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::tanh(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_log10(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_tanh(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


void Tensor::trunc(Tensor *A, Tensor *B){
    if (A->isCPU() && B->isCPU()) {
        cpu_trunc(A, B);
    }
#ifdef cGPU
    else if (B->isCPU() && B->isCPU())
      {
        gpu_trunc(A, B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}


// Math operations (binary) ************************
Tensor* Tensor::interpolate(float factor1, Tensor *A, float factor2, Tensor *B){
    Tensor* C = new Tensor(A->getShape(), A->device);
    Tensor::interpolate(factor1, A, factor2, B, C);
    return C;
}

void Tensor::interpolate(float factor1, Tensor *A, float factor2, Tensor *B, Tensor *C){
    Tensor::add(factor1, A, factor2, B, C, 1);
}


// ***** Overload operators *****************************
// Tensor and Tensor (Element wise)  ********************
Tensor& operator+ (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::add(&A, &B);
    return (*t);
}

//Tensor& operator- (Tensor &A, Tensor &B) {
//    Tensor* t = Tensor::sub(&A, &B);
//    return (*t);
//}

Tensor& operator* (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::mult(&A, &B);
    return (*t);
}


Tensor& operator/ (Tensor &A, Tensor &B) {
    Tensor* t = Tensor::div(&A, &B);
    return (*t);
}


// Tensor op= Tensor (Element wise)  ********************
void operator+= (Tensor &A, Tensor &B) {
    Tensor::add(1.0f, &A, 1.0f, &B, &A, 0);
}

void operator-= (Tensor &A, Tensor &B) {
    Tensor::add(1.0f, &A, -1.0f, &B, &A, 0);
}

void operator*= (Tensor &A, Tensor &B) {
    Tensor::el_mult(&A, &B, &A, 0);
}

void operator/= (Tensor &A, Tensor &B) {
    Tensor::el_div(&A, &B, &A, 0);
}

// Tensor op= Scalar  ********************
void operator+= (Tensor &A, float v) {
    A.add_(v);
}

void operator-= (Tensor &A, float v) {
    A.sub_(v);
}

void operator*= (Tensor &A, float v) {
    A.mult_(v);
}

void operator/= (Tensor &A, float v) {
    A.div_(v);
}

// Tensor and scalar *******************
Tensor& operator+ (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->add_(v);
    return (*t);
}

Tensor& operator- (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->add_(-v);
    return (*t);
}

Tensor& operator* (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->mult_(v);
    return (*t);
}

Tensor& operator/ (Tensor &A, float v) {
    Tensor* t = A.clone();
    t->div_(v);
    return (*t);
}


// Scalar and Tensor *******************
Tensor& operator+ (float v, Tensor &A) {
    return A + v;
}

Tensor& operator- (float v, Tensor &A) {
    Tensor* t = A.clone();
    t->neg_();
    t->add_(v);
    return (*t);
}

Tensor& operator* (float v, Tensor &A) {
    return A * v;
}

Tensor& operator/ (float v, Tensor &A) {
    Tensor* t = A.clone();
    t->inv_(v);
    return (*t);
}







void Tensor::add_(float v) {
    if (isCPU()) {
        cpu_add_(this, v);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_add_(this, v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void Tensor::add_(Tensor *A){
    Tensor::add(1.0f, this, 1.0f, A, this, 0);
}

Tensor* Tensor::add(Tensor *A, Tensor *B){
    auto *t_new = new Tensor(A->shape, A->device);
    add(1.0f, A, 1.0f, B, t_new, 0);
    return t_new;
}

void Tensor::add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// sum C=(sca*A)+(scb*B)
    //// or C+=(sca*A)+(scb*B) if incC is 1
    //// Dimensions and types must be compatible
    ///////////////////////////////////////
    int aux = 0;


    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::add_");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::add");
    }

    C->tsem->lock();
    if (A->isCPU()) {
        cpu_add(scA, A, scB, B, C, incC);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_addc(scA,A,scB,B,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    C->tsem->unlock();
}
void Tensor::add(Tensor *A, Tensor *B, Tensor *C) {
    Tensor::add(1.0, A, 1.0, B, C, 0);
}

void Tensor::inc(Tensor *A, Tensor *B) {
    // TODO: Review against add

    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::inc");


    if ((A->isCPU()) && (B->isCPU())) {
        cpu_inc(A, B);
    }
#ifdef cGPU
        else if ((A->isGPU())&&(B->isGPU())) {
        Tensor::add(1,A,1,B,B,0);
    }
    else if (((A->isCPU())&&(B->isGPU()))||((A->isGPU())&&(B->isCPU())))
    {
        Tensor *n=new Tensor(B->getShape(),B->device);
        Tensor::copy(A,n);
        Tensor::add(1,n,1,B,B,0);
        delete n;
    }
#endif
    else {
        fprintf(stderr, "(%d %d)\n", A->device, B->device);
        msg("unsupported inc between devices", "Tensor::inc");
    }
}


void Tensor::div_(float v) { mult_(1.0f / v); }

void Tensor::inv_(float v) {
    if (isCPU()) {
        cpu_inv_(this, v);
    }
#ifdef cGPU
    else if (isGPU())
    {
      gpu_inv_(this, v);
    }
#endif
#ifdef cFPGA
    else {

  }
#endif
}


Tensor* Tensor::div(Tensor *A, float v){
    auto *t_new = A->clone();
    t_new->div_(v);
    return t_new;
}

Tensor* Tensor::div(Tensor *A, Tensor *B){
    auto *t_new = new Tensor(A->shape, A->device);
    Tensor::el_div(A, B, t_new, 0);
    return t_new;
}

void Tensor::el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// Element Div C=A./B
    //// incC 1 means C+=A./B (increment over C)
    //// Dimensions must be compatible
    ///////////////////////////////////////

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::el_div");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) msg("Incompatible dims", "Tensor::el_div");

    C->tsem->lock();
    if (A->isCPU()) {
        cpu_el_div(A, B, C, incC);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_el_div(A,B,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}

void Tensor::exp_() {
    if (isCPU()) {
        cpu_exp_(this);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_exp_(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

float Tensor::max(){
    if (isCPU()) {
        return cpu_max(this);
    }
#ifdef cGPU
    else if (isGPU())
      {
        return gpu_max(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    return -1.0f;  // Temp
}

float Tensor::min(){
    if (isCPU()) {
        return cpu_min(this);
    }
#ifdef cGPU
    else if (isGPU())
      {
        return gpu_min(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    return -1.0f;  // Temp
}


void Tensor::mult_(float v) {
    if (isCPU()) {
        cpu_mult_(this, v);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_mult_(this, v);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::mult(Tensor *A, float v){
    auto *t_new = A->clone();
    t_new->mult_(v);
    return t_new;
}


Tensor* Tensor::mult(Tensor *A, Tensor *B){
    auto *t_new = new Tensor(A->shape, A->device);
    Tensor::el_mult(A, B, t_new, 0);
    return t_new;
}


void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// MULT2D C=A*B
    //// tA means transpose A {0,1}
    //// tB means transpose B {0,1}
    //// tC 1 means C+=A*B (increment over C)
    //// Dimensions and types must be compatible
    //// Only for 2D Tensors
    ///////////////////////////////////////

    if ((A->device != B->device) || (A->device != C->device)) {A->info();B->info();C->info();msg("Tensors in different devices", "Tensor::mult2D");}
    if ((A->ndim != 2) || (B->ndim != 2) || (C->ndim != 2)) msg("Only 2D tensors", "Tensor::mult2D");
    if (!tA) {
        if (!tB) {
            if ((A->shape[1] != B->shape[0]) || (A->shape[0] != C->shape[0]) || (B->shape[1] != C->shape[1]))
                msg("Incompatible dims", "Tensor::mult2D");
        } else if ((A->shape[1] != B->shape[1]) || (A->shape[0] != C->shape[0]) || (B->shape[0] != C->shape[1]))
            msg("Incompatible dims", "Tensor::mult2D");
    } else {
        if (!tB) {
            if ((A->shape[0] != B->shape[0]) || (A->shape[1] != C->shape[0]) || (B->shape[1] != C->shape[1]))
                msg("Incompatible dims", "Tensor::mult2D");
        } else if ((A->shape[0] != B->shape[1]) || (A->shape[1] != C->shape[0]) || (B->shape[0] != C->shape[1]))
            msg("Incompatible dims", "Tensor::mult2D");
    }


    C->tsem->lock();

    if (A->isCPU()) {
        cpu_mult2D(A, tA, B, tB, C, incC);
    }

#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_mult2D(A,tA,B,tB,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


void Tensor::el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// Element Mult C=A.*B
    //// incC 1 means C+=A.*B (increment over C)
    //// Dimensions must be compatible
    ///////////////////////////////////////
    C->tsem->lock();
    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::el_mult");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) {
        A->info();
        B->info();
        C->info();
        msg("Incompatible dims", "Tensor::el_mult");
    }

    if (A->isCPU()) {
        cpu_el_mult(A, B, C, incC);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_el_mult(A,B,C,incC);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


float Tensor::sum() {
    if (isCPU()) {
        return cpu_sum(this);
    }
#ifdef cGPU
    else if (isGPU())
    {
        return gpu_sum(this);
    }
#endif
#ifdef cFPGA
    else {

    }
#endif
    return 0;
}

//Tensor* Tensor::sum(Tensor *A){}

void Tensor::sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
    ///////////////////////////////////////
    //// sum2D_rowise C=A.rowise+B
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    ///////////////////////////////////////
    if ((A->device != B->device) || (A->device != C->device))
        msg("Tensors in different devices", "Tensor::sum2D_rowwise");
    if ((A->ndim != 2) || (B->ndim != 1) || (C->ndim != 2)) msg("sum2D_rowwise dims");
    if ((!eqsize(A, C)) || (A->shape[1] != B->shape[0])) msg("Incompatible dims", "Tensor::sum2D_rowwise");

    C->tsem->lock();
    if (A->isCPU()) {
        cpu_sum2D_rowwise(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum2D_rowwise(A,B,C);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}


void Tensor::reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
    ///////////////////////////////////////
    //// reduce_sum2D B=reduce_sum2D(A)
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    //// axis is the dimension to be sumed
    ///////////////////////////////////////
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::reduce_sum2D");
    if ((A->ndim - 1) != B->ndim) msg("Incorrect dims", "Tensor::reduce_sum2D");
    if ((A->shape[1 - axis] != B->shape[0])) msg("Incompatible dims", "Tensor::reduce_sum2D");

    B->tsem->lock();
    if (A->isCPU()) {
        cpu_reduce_sum2D(A, B, axis, incB);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_reduce_sum2D(A,B,axis,incB);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();
}

void Tensor::sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
    ///////////////////////////////////////
    //// sum2D_colwise C=A.colwise+B
    //// Dimensions and types must be compatible
    //// A is 2D Tensor
    //// B is 1D Tensor
    ///////////////////////////////////////
    if ((A->device != B->device) || (A->device != C->device))
        msg("Tensors in different devices", "Tensor::sum2D_colwise");
    if ((A->ndim != 2) || (B->ndim != 1) || (C->ndim != 2)) msg("sum2D_colwise dims");
    if ((!eqsize(A, C)) || (A->shape[0] != B->shape[0])) msg("Incompatible dims", "Tensor::sum2D_colwise");

    C->tsem->lock();
    if (A->isCPU()) {
        cpu_sum2D_colwise(A, B, C);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum2D_colwise(A,B,C);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}

float Tensor::sum_abs() {
    if (isCPU()) {
        return cpu_sum_abs(this);
    }
#ifdef cGPU
    else if (isGPU())
      {
         //return gpu_sum_abs(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    return 0;
}
