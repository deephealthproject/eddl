/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "constraint.h"

using namespace std;


CMaxNorm::CMaxNorm(float max_value, int axis) : Constraint("max_norm") {
    // Todo: Implement
    this->max_value;
    this->axis;
}

float CMaxNorm::apply(Tensor* T) { return 0; }
