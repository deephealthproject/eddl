/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/regularizers/regularizer.h"

using namespace std;


RL2::RL2(float l2) : Regularizer("l2") {
    this->l2 = l2;
}

RL2::~RL2()= default;

void RL2::apply(Tensor* T) {
    Tensor *B = T->clone();

    Tensor::add(1.0f, T, -this->l2, B, T, 0);

    delete B;
}
