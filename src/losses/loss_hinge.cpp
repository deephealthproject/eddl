/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
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

LHinge::LHinge() : Loss("hinge"){}

void LHinge::delta(Tensor *T, Tensor *Y, Tensor *D) {
}

float LHinge::value(Tensor *T, Tensor *Y) {
    return 0;
}
Loss* LHinge::clone()
{
  return new LHinge();
}
