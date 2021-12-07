/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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
