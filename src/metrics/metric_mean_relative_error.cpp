/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/metrics/metric.h"

using namespace std;

MMeanRelativeError::MMeanRelativeError(float eps) : Metric("mean_relative_error")
{
  this->eps=eps;
}

float MMeanRelativeError::value(Tensor *T, Tensor *Y) {
    float f;
    int size=T->size/T->shape[0];  // batch is divided in print_loss

    // batch error: add((T-Y)^2)
    Tensor *aux1 = new Tensor(T->getShape(), T->device);
    Tensor *aux2 = T->clone();

    Tensor::add(1.0, T, -1.0, Y, aux1, 0);
    aux1->abs_();


    aux2->abs_();
    aux2->add_(eps);

    Tensor::el_div(aux1,aux2,aux1,0);


    f = aux1->sum()/size;


    delete aux1;
    delete aux2;
    return f;
}

Metric* MMeanRelativeError::clone() {
  return new MMeanRelativeError();
}
