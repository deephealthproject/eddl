/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "regularizer.h"

using namespace std;


RL1L2::RL1L2(float l1, float l2) : Regularizer("l1_l2") {
    this->l1 = l1;
    this->l2 = l2;
}

void RL1L2::apply(Tensor* T) {
  Tensor *S = T->clone();
  Tensor *B = T->clone();
  Tensor *A = T->clone();

  S->sign_();
  Tensor::el_mult(S,B,B,0);
  Tensor::add(1.0f, T, -this->l1, B, T, 0);

  Tensor::add(1.0f, T, -this->l2, A, T, 0);

  delete A;
  delete B;
  delete S;
}
