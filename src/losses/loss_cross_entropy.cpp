/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/losses/loss.h"

using namespace std;


LCrossEntropy::LCrossEntropy() : Loss("cross_entropy"){}

void LCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    float eps=0.000001;
    Tensor *aux1;
    Tensor *aux2;
    Tensor *one;

    // delta: t/y - (1-t)/(1-y)
    aux1 = new Tensor(T->getShape(), T->device);
    aux2 = new Tensor(T->getShape(), T->device);
    one = new Tensor(T->getShape(), T->device);

    one->fill_(1.0);

    //  (1-t)/(1-y)
    Tensor::add(1, one, -1, T, aux1, 0);
    Tensor::add(1, one, -1, Y, aux2, 0);
    aux2->add_(eps);
    Tensor::el_div(aux1, aux2, aux2, 0);

    // t/y
    Y->add_(eps);
    Tensor::el_div(T, Y, aux1, 0);


    Tensor::add(-1, aux1, 1, aux2, D, 0);
    D->div_(D->shape[0]);

    delete aux1;
    delete aux2;
    delete one;
}

float LCrossEntropy::value(Tensor *T, Tensor *Y) {
    float f;
    Tensor *aux1;

    aux1 = new Tensor(T->getShape(), T->device);
    cent(T, Y, aux1);
    f = aux1->sum();

    delete aux1;

    return f;
}
