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


RL1::RL1(float l1) : Regularizer("l1") {
    this->l1 = l1;
}

void RL1::apply(Tensor* T) {
    Tensor *B = T->clone();

    B->abs_();
    Tensor::add(1.0f, T, this->l1, B, T, 0.0);

    delete B;
}
