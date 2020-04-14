/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/losses/loss.h"

using namespace std;


LMeanSquaredError::LMeanSquaredError() : Loss("mean_squared_error"){}

void LMeanSquaredError::delta(Tensor *T, Tensor *Y, Tensor *D) {
    //delta: (Y-T)
    Tensor::add(-1.0, T, 1.0, Y, D, 0);
    D->div_(D->shape[0]);
}

float LMeanSquaredError::value(Tensor *T, Tensor *Y) {
    float f;
    // batch error: add((T-Y)^2)
    Tensor *aux1;
    int size=T->size/T->shape[0];  // batch is divided in print_loss

    aux1 = new Tensor(T->getShape(), T->device);
    Tensor::add(1.0, T, -1.0, Y, aux1, 0);
    Tensor::el_mult(aux1, aux1, aux1, 0);
    f = aux1->sum()/size;

    delete aux1;

    return f;
}
