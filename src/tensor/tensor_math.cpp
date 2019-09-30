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

#include "tensor.h"
#include <cmath>
#include <limits>

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

#define MAX_FLOAT std::numeric_limits<float>::max()
#define MIN_FLOAT -std::numeric_limits<float>::max()
#define PRECISION_FLOAT -std::numeric_limits<float>::max()

using namespace std;


void Tensor::abs_() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::fabs(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::abs(Tensor *A){}

void Tensor::acos_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::acosf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::acos(Tensor *A){}

void Tensor::add_(float v) {
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

Tensor* Tensor::add(Tensor *A, Tensor *B){}

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

        for (int i = 0; i < A->size; i++)
            if (incC) C->ptr[i] += scA * A->ptr[i] + scB * B->ptr[i];
            else C->ptr[i] = scA * A->ptr[i] + scB * B->ptr[i];
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_sum(scA,A,scB,B,C,incC);
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
        B->tsem->lock();

        for (int i = 0; i < A->size; i++)
            B->ptr[i] += A->ptr[i];

        B->tsem->unlock();
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

void Tensor::asin_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::asinf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::asin(Tensor *A){}

void Tensor::atan_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::atanf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::atan(Tensor *A){}

void Tensor::ceil_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::ceilf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::ceil(Tensor *A){}

void Tensor::clamp_(float min, float max){
    if (isCPU()) {
        for (int i = 0; i < size; ++i){
            if (ptr[i] < min){
                ptr[i] = min;
            } else if(ptr[i] > max){
                ptr[i] = max;
            }
        }
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::clamp(Tensor *A, float min, float max){}

void Tensor::clampmax_(float max){ clamp_(MIN_FLOAT, max); }
Tensor* Tensor::clampmax(Tensor *A, float max){}

void Tensor::clampmin_(float min){ clamp_(min, MAX_FLOAT); }
Tensor* Tensor::clampmin(Tensor *A, float min){}

void Tensor::cos_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::cosf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::cos(Tensor *A){}

void Tensor::cosh_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::coshf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::cosh(Tensor *A){}


void Tensor::div_(float v) { mult_(1.0f / v); }

Tensor* Tensor::div(Tensor *A){}

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

        for (int i = 0; i < A->size; i++)
            if (incC) C->ptr[i] += A->ptr[i] / B->ptr[i];
            else C->ptr[i] = A->ptr[i] / B->ptr[i];
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
        for (int i = 0; i < size; ++i) ptr[i] = ::expf(ptr[i]);
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


Tensor* Tensor::exp(Tensor *A){}

void Tensor::floor_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::floorf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::floor(Tensor *A){}


void Tensor::log_() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::logf(ptr[i]);
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

Tensor* Tensor::log(Tensor *A){}

void Tensor::log2_() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::log2f(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log2(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::log2(Tensor *A){}


void Tensor::log10_() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::log10f(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log10(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::log10(Tensor *A){}


void Tensor::logn_(float n) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::logf(ptr[i])/::logf(n);
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

Tensor* Tensor::logn(Tensor *A){}

float Tensor::max(){
    if (isCPU()) {
        float max = MIN_FLOAT;
        for (int i = 0; i < size; ++i) {
            if (ptr[i] > max) { max = ptr[i]; }
        }
        return max;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log10(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    return -1.0f;  // Temp
}

float Tensor::max(Tensor *A){}

float Tensor::min(){
    if (isCPU()) {
        float min = MAX_FLOAT;
        for (int i = 0; i < size; ++i) {
            if (ptr[i] < min) { min = ptr[i]; }
        }
        return min;
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log10(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    return -1.0f;  // Temp
}

float Tensor::min(Tensor *A){}


void Tensor::mod_(float v){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::fmod(ptr[i], v);
    }
#ifdef cGPU
    else if (isGPU())
      {
        gpu_log10(this);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::mod(Tensor *A, float v){};

void Tensor::mult_(float v) {
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

Tensor* Tensor::mult(Tensor *A){}


void Tensor::mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    ///////////////////////////////////////
    //// MULT2D C=A*B
    //// tA means transpose A {0,1}
    //// tB means transpose B {0,1}
    //// tC 1 means C+=A*B (increment over C)
    //// Dimensions and types must be compatible
    //// Only for 2D Tensors
    ///////////////////////////////////////

    if ((A->device != B->device) || (A->device != C->device)) msg("Tensors in different devices", "Tensor::mult2D");
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
        for (int i = 0; i < A->size; i++)
            if (incC) C->ptr[i] += A->ptr[i] * B->ptr[i];
            else C->ptr[i] = A->ptr[i] * B->ptr[i];
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

void Tensor::neg_(){ mult_(-1.0f); }

Tensor* Tensor::neg(Tensor *A){};


void Tensor::normalize_(float min, float max){
    // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
    // (max2-min2)/(max1-min1) * (x-min1) + min2
    float max_ori = this->max();
    float min_ori = this->min();
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = (max-min)/(max_ori-min_ori) * (ptr[i]-min_ori) + min;
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

static Tensor* normalize_(Tensor *A, float min=0.0f, float max=1.0f);


void Tensor::pow_(float exp) {
    // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
    // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
    // speed: 0.057887s
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::powf(ptr[i], exp);
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

Tensor* Tensor::pow(Tensor *A){}

void Tensor::reciprocal_() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = 1.0f/ptr[i];
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::reciprocal(Tensor *A){}

void Tensor::remainder_(float v) {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = (int)(ptr[i]/v);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::remainder(Tensor *A, float v){}

void Tensor::round_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::roundf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::round(Tensor *A){}

void Tensor::rsqrt_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = 1.0f/::sqrtf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::rsqrt(Tensor *A){}

void Tensor::sigmoid_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::expf(ptr[i])/(::expf(ptr[i])+1.0f);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::sigmoid(Tensor *A){}

void Tensor::sign_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) {
            if(ptr[i] > 0.0f){
                ptr[i] = 1.0f;
            }else if(ptr[i] < 0.0f){
                ptr[i] = -1.0f;
            }else{
                ptr[i] = 0.0f;
            }
        };
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::sign(Tensor *A){}

void Tensor::sign(Tensor *A, Tensor *B) {
    ///////////////////////////////////////
    /// Get sign (+-) of all values
    //////////////////////////////////////
    B->tsem->lock();

    if (!Tensor::eqsize(A, B))
        msg("Tensors with different shape", "Tensor::sign");
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::sign");

    if (A->isCPU()) {
        for (int i = 0; i < A->size; i++)
            if (A->ptr[i] < 0) B->ptr[i] = -1.0;
            else B->ptr[i] = 1.0;
    }
#ifdef cGPU
    else if (A->isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();

}

void Tensor::sin_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::sinf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::sin(Tensor *A){}

void Tensor::sinh_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::sinhf(ptr[i]);
    }
#ifdef cGPU
    else if (isGPU())
      {

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

Tensor* Tensor::sinh(Tensor *A){}


void Tensor::sqr_() {
    // pow(x, 2) == x*x  To know more, read comments in pow_'s function
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

Tensor* Tensor::sqr(Tensor *A){}


void Tensor::sqrt_() {
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::sqrtf(ptr[i]);
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

Tensor* Tensor::sqrt(Tensor *A){}


void Tensor::sub_(float v) { add_(-v); }

Tensor* Tensor::sub(Tensor *A, Tensor *B){}

float Tensor::sum_() {
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

Tensor* Tensor::sum(Tensor *A){}

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
        int p = 0;
        for (int i = 0; i < A->shape[0]; i++) {
            for (int j = 0; j < A->shape[1]; j++, p++)
                C->ptr[p] = A->ptr[p] + B->ptr[j];
        }
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
        int p = 0;
        for (int i = 0; i < A->shape[0]; i++) {
            for (int j = 0; j < A->shape[1]; j++, p++)
                C->ptr[p] = A->ptr[p] + B->ptr[i];
        }
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

float Tensor::sum_abs_() {
    if (isCPU()) {
        float sum = 0.0;
        for (int i = 0; i < size; ++i) sum += ::fabs(ptr[i]);
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

Tensor* Tensor::sum_abs(Tensor *A){}


void Tensor::tan_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::tanf(ptr[i]);
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

Tensor* Tensor::tan(Tensor *A){}

void Tensor::tanh_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = ::tanhf(ptr[i]);
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

Tensor* Tensor::tanh(Tensor *A){}


void Tensor::trunc_(){
    if (isCPU()) {
        for (int i = 0; i < size; ++i) ptr[i] = (int)(ptr[i]);
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

Tensor* Tensor::trunc(Tensor *A){}