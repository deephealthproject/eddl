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

#include "eddl/losses/loss.h"

using namespace std;

LMeanRelativeError::LMeanRelativeError() : Loss("mean_relative_error"){}

void LMeanRelativeError::delta(Tensor *T, Tensor *Y, Tensor *D) {
}

float LMeanRelativeError::value(Tensor *T, Tensor *Y) {
    return 0;
}
Loss* LMeanRelativeError::clone()
{
  return new LMeanRelativeError();
}
