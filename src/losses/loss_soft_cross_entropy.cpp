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

#include "loss.h"

using namespace std;


LSoftCrossEntropy::LSoftCrossEntropy() : Loss("soft_cross_entropy"){}

void LSoftCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    Tensor::add(1.0, T, -1.0, Y, D, 0);
}

float LSoftCrossEntropy::value(Tensor *T, Tensor *Y) {
    float f;
    Tensor *aux1;

    aux1 = new Tensor(T->getShape(), T->device);
    cent(T, Y, aux1);
    f = aux1->sum();

    delete aux1;

    return f;
}
