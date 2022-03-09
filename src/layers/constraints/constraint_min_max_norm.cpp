/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/layers/constraints/constraint.h"


CMinMaxNorm::CMinMaxNorm(float min_value, float max_value, float rate, int axis) : Constraint("min_max_norm") {
    // Todo: Implement
    // this->min_value;
    // this->max_value;
    // this->rate;
    // this->axis;
}

float CMinMaxNorm::apply(Tensor* T) { return 0; }
