/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/regularizers/regularizer.h"

using namespace std;


RL1L2::RL1L2(float l1, float l2) : Regularizer("l1_l2") {
    this->l1 = l1;
    this->l2 = l2;
}

RL1L2::~RL1L2()= default;

void RL1L2::apply(Tensor* T) {
  Tensor *S = T->clone();
  Tensor *A = T->clone();

  S->sign_();
  Tensor::add(1.0f, T, -this->l1, S, T, 0);

  Tensor::add(1.0f, T, -this->l2, A, T, 0);

  delete A;
  delete S;
}
