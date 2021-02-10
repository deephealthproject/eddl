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

#include "eddl/losses/loss.h"

using namespace std;

LMeanAbsoluteError::LMeanAbsoluteError() : Loss("mean_absolute_error"){}

void LMeanAbsoluteError::delta(Tensor *T, Tensor *Y, Tensor *D) {
}

float LMeanAbsoluteError::value(Tensor *T, Tensor *Y) {
    return 0;
}
Loss* LMeanAbsoluteError::clone()
{
  return new LMeanAbsoluteError();
}
