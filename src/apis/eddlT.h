/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _EDDLT_
#define _EDDLT_


#include "../tensor/tensor.h"

namespace eddlT{

  #define tensor Tensor*

  // Creation ops ***********************************
  Tensor* create(const vector<int> &shape, float *ptr=nullptr, int dev=DEV_CPU);
  Tensor* create(const vector<int> &shape, int dev=DEV_CPU);
  Tensor* create(const vector<int> &shape, float *ptr=nullptr);

  Tensor* zeros(const vector<int> &shape, int dev=DEV_CPU);
  Tensor* ones(const vector<int> &shape, int dev=DEV_CPU);
  Tensor* full(const vector<int> &shape, float value, int dev=DEV_CPU);
  Tensor* arange(float start, float end, float step=1.0f, int dev=DEV_CPU);
  Tensor* range(float start, float end, float step=1.0f, int dev=DEV_CPU);
  Tensor* linspace(float start, float end, int steps=100, int dev=DEV_CPU);
  Tensor* logspace(float start, float end, int steps=100, float base=10.0f, int dev=DEV_CPU);
  Tensor* eye(int size, int dev=DEV_CPU);
  Tensor* randn(const vector<int> &shape, int dev=DEV_CPU);

  // Pointer functions ********************************
  float *getptr(Tensor *t);

  // Load from file ***********************************
  Tensor *load(string fname);

  // Math ops       ***********************************
  void div_(Tensor *t,float f);

}

#endif
