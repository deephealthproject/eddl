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

#include "metric.h"

using namespace std;

MMeanSquaredError::MMeanSquaredError() : Metric("mean_squared_error"){}

float MMeanSquaredError::value(Tensor *T, Tensor *Y) {
    float f;
    // batch error: add((T-Y)^2)
    int size=T->size/T->shape[0];

    Tensor *aux1 = new Tensor(T->getShape(), T->device);

    Tensor::add(1.0, T, -1.0, Y, aux1, 0);
    Tensor::el_mult(aux1, aux1, aux1, 0);
    f = aux1->sum()/size;

    delete aux1;
    return f;
}
