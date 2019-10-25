/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddlT.h"


using namespace std;

////////////////////////////////////////////////////////
///// EDDLT is a wrapper class to deal with Tensors
////////////////////////////////////////////////////////
namespace eddlT {


  // Creation ops ***********************************
  Tensor* create(const vector<int> &shape, int dev)
  {
    return new Tensor(shape,dev);
  }
  Tensor* create(const vector<int> &shape, float *ptr)
  {
    return new Tensor(shape,ptr);
  }
  Tensor* create(const vector<int> &shape)
  {
    return new Tensor(shape);
  }

  Tensor* zeros(const vector<int> &shape, int dev)
  {
    return Tensor::zeros(shape,dev);
  }
  Tensor* ones(const vector<int> &shape, int dev){
    return Tensor::ones(shape,dev);
  }
  Tensor* full(const vector<int> &shape, float value, int dev){
    return Tensor::full(shape,value,dev);
  }
  Tensor* arange(float start, float end, float step, int dev){
    return Tensor::arange(start,end,step,dev);
  }
  Tensor* range(float start, float end, float step, int dev){
    return Tensor::range(start,end,step,dev);
  }
  Tensor* linspace(float start, float end, int steps, int dev){
    return Tensor::linspace(start,end,steps,dev);
  }
  Tensor* logspace(float start, float end, int steps, float base, int dev){
    return Tensor::logspace(start,end,steps,dev);
  }
  Tensor* eye(int size, int dev){
    return Tensor::eye(size,dev);
  }

  Tensor* randn(const vector<int> &shape, int dev){
    return Tensor::randn(shape,dev);
  }

  // Copy data        ********************************
  void ToCPU_(Tensor *A)
  {
    A->ToCPU();
  }
  void ToGPU_(Tensor *A)
  {
    A->ToGPU();
  }
  Tensor * ToCPU(Tensor *A){
    Tensor *B=A->clone();
    B->ToCPU();
    return B;
  }
  Tensor * ToGPU(Tensor *A)
  {
    Tensor *B=A->clone();
    B->ToGPU();
    return B;
  }
  Tensor* clone(Tensor *A)
  {
    return A->clone();
  }


  // Pointer functions ********************************
  float *getptr(Tensor *A){
    return A->ptr;
  }

  // Print functions   ********************************
  void print(Tensor *A){
    A->print();
  }
  void info(Tensor *A)
  {
    A->info();
  }

  // Load from file ***********************************
  Tensor *load(string fname){
    FILE *fe = fopen(fname.c_str(), "rb");
    if (fe == nullptr) {
        fprintf(stderr, "%s not found\n", fname.c_str());
        exit(1);
    }

    vector<int> shape;
    int ndim,v;
    int read = fread(&ndim, sizeof(int), 1, fe);
    for (int i = 0; i < ndim; ++i) {
        int read = fread(&v, sizeof(int), 1, fe);
        shape.push_back(v);
    }

    Tensor *t=new Tensor(shape,DEV_CPU);
    t->load(fe);
    return t;
  }


  // Math ops       ***********************************

  void abs_(Tensor *A) {
    A->abs_();
  }
  Tensor* abs(Tensor *A){
    return Tensor::abs(A);
  }

  void acos_(Tensor *A){
    A->acos_();
  }
  Tensor* acos(Tensor *A){
    return Tensor::acos(A);
  }

  void add_(Tensor *A,float v)
  {
    A->add_(v);
  }
  Tensor *add(Tensor *A,float v){
    Tensor *B=A->clone();
    B->add_(v);
    return B;
  }
  void add_(Tensor *A,Tensor *B){
    Tensor::inc(B,A);
  }
  Tensor* add(Tensor *A, Tensor *B)
  {
    return Tensor::add(A,B);
  }

  void inc_(Tensor *A,Tensor *B){
    Tensor::inc(A,B);
  }

  void asin_(Tensor *A){
    A->asin_();
  }
  Tensor* asin(Tensor *A){
    return Tensor::asin(A);
  }

  void atan_(Tensor *A){
    A->atan_();
  }
  Tensor* atan(Tensor *A){
    return Tensor::atan(A);
  }
  void ceil_(Tensor *A){
    A->ceil_();
  }
  Tensor* ceil(Tensor *A){
    return Tensor::ceil(A);
  }

  void clamp_(Tensor *A,float min, float max){
    A->clamp_(min,max);
  }
  Tensor* clamp(Tensor *A,float min, float max){
    return Tensor::clamp(A, min,  max);
  }

  void clampmax_(Tensor *A,float max){
    A->clampmax_(max);
  }
  Tensor* clampmax(Tensor *A,float max){
    return Tensor::clampmax(A,  max);
  }

  void clampmin_(Tensor *A,float min){
    A->clampmin_(min);
  }
  Tensor* clampmin(Tensor *A,float min){
    return Tensor::clampmin(A, min);
  }

  void cos_(Tensor *A){
    A->cos_();
  }
  Tensor* cos(Tensor *A){
    return Tensor::cos(A);
  }

  void cosh_(Tensor *A){
    A->cosh_();
  }
  Tensor* cosh(Tensor *A){
    return Tensor::cosh(A);
  }

  void div_(Tensor *A,float v)
  {
    A->div_(v);
  }
  Tensor *div(Tensor *A,float v){
    Tensor *B=A->clone();
    B->div_(v);
    return B;
  }
  void div_(Tensor *A,Tensor *B){
    Tensor::el_div(A,B,A,0);
  }
  Tensor* div(Tensor *A, Tensor *B)
  {
    Tensor *C=new Tensor(A->getShape(),A->device);
    Tensor::el_div(A,B,C,0);
    return C;
  }


  void exp_(Tensor *A){
    A->exp_();
  }
  Tensor* exp(Tensor *A){
    return Tensor::exp(A);
  }


  void floor_(Tensor *A){
    A->floor_();
  }
  Tensor* floor(Tensor *A){
    return Tensor::floor(A);
  }


   void log_(Tensor *A){
     A->log_();
   }
   Tensor* log(Tensor *A){
     return Tensor::log(A);
   }

   void log2_(Tensor *A){
     A->log2_();
   }
   Tensor* log2(Tensor *A){
     return Tensor::log2(A);
   }

   void log10_(Tensor *A){
     A->log10_();
   }
   Tensor* log10(Tensor *A){
     return Tensor::log10(A);
   }

  void logn_(Tensor *A,float n){
    A->logn_(n);
  }
  Tensor* logn(Tensor *A, float n)
  {
    Tensor *B=A->clone();
    B->logn_(n);
    return B;
  }


  float max(Tensor *A){
    return A->max();
  }

  float min(Tensor *A){
    return A->min();
  }

  void mod_(Tensor *A,float v){
    A->mod_(v);
  }
  Tensor* mod(Tensor *A,float v){
    return Tensor::mod(A,v);
  }


  void mult_(Tensor *A,float v)
  {
    A->mult_(v);
  }
  Tensor *mult(Tensor *A,float v){
    Tensor *B=A->clone();
    B->mult_(v);
    return B;
  }
  void mult_(Tensor *A,Tensor *B){
    Tensor::el_mult(A,B,A,0);
  }
  Tensor* mult(Tensor *A, Tensor *B)
  {
    Tensor *C=new Tensor(A->getShape(),A->device);
    Tensor::el_mult(A,B,C,0);
    return C;
  }


   Tensor *mult2D(Tensor *A, Tensor *B)
   {

     Tensor *C=new Tensor({A->shape[0],B->shape[1]},A->device);
     Tensor::mult2D(A,0,B,0,C,0);
     return C;
   }


   void neg_(Tensor *A){
     A->neg_();
   }
   Tensor* neg(Tensor *A){
     return Tensor::neg(A);
   }


  void normalize_(Tensor *A,float min, float max){
    A->normalize_(min,max);
  }

  Tensor* normalize(Tensor *A, float min, float max)
  {
    Tensor *B=A->clone();
    B->normalize_(min,max);
    return B;

  }

  void pow_(Tensor *A,float exp);
   Tensor* pow(Tensor *A, float exp);


   void reciprocal_(Tensor *A){
     A->reciprocal_();
   }
   Tensor* reciprocal(Tensor *A){
     return Tensor::reciprocal(A);
   }

  void remainder_(Tensor *A,float v);
   Tensor* remainder(Tensor *A, float v);


   void round_(Tensor *A){
     A->round_();
   }
   Tensor* round(Tensor *A){
     return Tensor::round(A);
   }


   void rsqrt_(Tensor *A){
     A->rsqrt_();
   }
   Tensor* rsqrt(Tensor *A){
     return Tensor::rsqrt(A);
   }

   void sigmoid_(Tensor *A){
     A->sigmoid_();
   }
   Tensor* sigmoid(Tensor *A){
     return Tensor::sigmoid(A);
   }



   void sign_(Tensor *A){
     A->sign_();
   }
   Tensor* sign(Tensor *A){
     return Tensor::sign(A);
   }

  void sin_(Tensor *A){
    A->sin_();
  }
  Tensor* sin(Tensor *A){
    return Tensor::sin(A);
  }

   void sinh_(Tensor *A){
     A->sinh_();
   }
   Tensor* sinh(Tensor *A){
     return Tensor::sinh(A);
   }

   void sqr_(Tensor *A){
     A->sqr_();
   }
   Tensor* sqr(Tensor *A){
     return Tensor::sqr(A);
   }
   void sqrt_(Tensor *A){
     A->sqrt_();
   }
   Tensor* sqrt(Tensor *A){
     return Tensor::sqrt(A);
   }

   void sub_(Tensor *A,float v)
   {
     A->sub_(v);
   }
   Tensor *sub(Tensor *A,float v){
     Tensor *B=A->clone();
     B->sub_(v);
     return B;
   }
   void sub_(Tensor *A,Tensor *B){
     Tensor::add(1.0,A,-1.0,B,A,0);
   }
   Tensor* sub(Tensor *A, Tensor *B)
   {
     return Tensor::sub(A,B);
   }

  float sum(Tensor *A);
  float sum_abs(Tensor *A);


  void tan_(Tensor *A){
    A->tan_();
  }
  Tensor* tan(Tensor *A){
    return Tensor::tan(A);
  }


  void tanh_(Tensor *A){
    A->tanh_();
  }
  Tensor* tanh(Tensor *A){
    return Tensor::tanh(A);
  }


  void trunc_(Tensor *A){
    A->trunc_();
  }
  Tensor* trunc(Tensor *A){
    return Tensor::trunc(A);
  }



}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
