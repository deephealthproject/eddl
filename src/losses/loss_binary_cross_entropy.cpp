/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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


LBinaryCrossEntropy::LBinaryCrossEntropy() : Loss("binary_cross_entropy"){}

void LBinaryCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    tensorNN::d_binary_cross_entropy(T, Y, D);
    D->div_(D->shape[0]);
}

float LBinaryCrossEntropy::value(Tensor *T, Tensor *Y) {
    float loss_value = tensorNN::binary_cross_entropy(T, Y);
    return loss_value;
}

Loss* LBinaryCrossEntropy::clone()
{
  return new LBinaryCrossEntropy();
}
