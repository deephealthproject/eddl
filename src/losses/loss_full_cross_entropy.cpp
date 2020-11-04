/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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


LFullCrossEntropy::LFullCrossEntropy() : Loss("full_cross_entropy"){}

void LFullCrossEntropy::delta(Tensor *T, Tensor *Y, Tensor *D) {
    tensorNN::D_FullCrossEntropy(T, Y, D);
}

float LFullCrossEntropy::value(Tensor *T, Tensor *Y) {
    float loss_value = tensorNN::FullCrossEntropy(T, Y);
    return loss_value;
}

Loss* LFullCrossEntropy::clone()
{
  return new LFullCrossEntropy();
}
