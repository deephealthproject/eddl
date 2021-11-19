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
