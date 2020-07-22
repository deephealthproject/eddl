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


RL1::RL1(float l1) : Regularizer("l1") {
    this->l1 = l1;
}

RL1::~RL1()= default;;

void RL1::apply(Tensor* T) {
    Tensor *S = T->clone();

    S->sign_();
    Tensor::add(1.0f, T, -this->l1, S, T, 0);

    delete S;
}
