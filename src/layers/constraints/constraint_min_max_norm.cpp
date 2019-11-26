/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "constraint.h"


CMinMaxNorm::CMinMaxNorm(float min_value, float max_value, float rate, int axis) : Constraint("min_max_norm") {
    // Todo: Implement
    this->min_value;
    this->max_value;
    this->rate;
    this->axis;
}

float CMinMaxNorm::apply(Tensor* T) { return 0; }
