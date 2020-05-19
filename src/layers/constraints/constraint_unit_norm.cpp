/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/layers/constraints/constraint.h"

CUnitNorm::CUnitNorm(int axis) : Constraint("unit_norm") {
    // Todo: Implement
    // this->axis;
}

float CUnitNorm::apply(Tensor* T) { return 0; }
