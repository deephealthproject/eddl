/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/layers/constraints/constraint.h"

using namespace std;


CMaxNorm::CMaxNorm(float max_value, int axis) : Constraint("max_norm") {
    // Todo: Implement
    //this->max_value;
    //this->axis;
}

float CMaxNorm::apply(Tensor* T) { return 0; }
