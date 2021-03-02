/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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


LCategoricalCrossEntropy::LCategoricalCrossEntropy() : Loss("categorical_cross_entropy"){}

void LCategoricalCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    tensorNN::d_categorical_cross_entropy(T, Y, D);
    D->div_(D->shape[0]);
}

float LCategoricalCrossEntropy::value(Tensor *T, Tensor *Y) {
    float loss_value = tensorNN::categorical_cross_entropy(T, Y);
    return loss_value;
}

Loss* LCategoricalCrossEntropy::clone()
{
  return new LCategoricalCrossEntropy();
}
