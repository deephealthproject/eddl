/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/metrics/metric.h"

using namespace std;

MMeanAbsoluteError::MMeanAbsoluteError() : Metric("mean_absolute_error"){}

float MMeanAbsoluteError::value(Tensor *T, Tensor *Y) {
    float f;
    int size=T->size/T->shape[0];  // batch is divided in print_loss

    // batch error: add((T-Y)^2)
    auto *aux1 = new Tensor(T->getShape(), T->device);

    Tensor::add(1.0, T, -1.0, Y, aux1, 0);

    aux1->abs_();

    f = aux1->sum()/size;


    delete aux1;
    return f;
}

Metric* MMeanAbsoluteError::clone() {
  return new MMeanAbsoluteError();
}
