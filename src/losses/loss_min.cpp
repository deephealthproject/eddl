/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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


LMin::LMin() : Loss("min"){}

void LMin::delta(Tensor *T, Tensor *Y, Tensor *D) {
    D->fill_(1);
}

float LMin::value(Tensor *T, Tensor *Y) {
    float sum;
    sum=Y->sum();
    return sum;
}
Loss* LMin::clone()
{
  return new LMin();
}
