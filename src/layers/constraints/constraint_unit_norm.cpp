/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/layers/constraints/constraint.h"

CUnitNorm::CUnitNorm(int axis) : Constraint("unit_norm") {
    // Todo: Implement
    // this->axis;
}

float CUnitNorm::apply(Tensor* T) { return 0; }
