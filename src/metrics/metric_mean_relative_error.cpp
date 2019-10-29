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

MMeanRelativeError::MMeanRelativeError() : Metric("mean_relative_error"){}

float MMeanRelativeError::value(Tensor *T, Tensor *Y) {
    float f;

    // batch error: add((T-Y)^2)
    Tensor *aux1 = new Tensor(T->getShape(), T->device);
    Tensor *aux2 = T->clone();

    Tensor::add(1.0, T, -1.0, Y, aux1, 0);


    aux1->abs_();


    aux2->abs_();
    aux2->add_(1.0);

    Tensor::el_div(aux1,aux2,aux1,0);


    f = aux1->sum()/T->shape[0];


    delete aux1;
    delete aux2;
    return f;
}
