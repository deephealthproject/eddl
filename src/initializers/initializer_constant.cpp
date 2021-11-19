/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/initializers/initializer.h"

using namespace std;

/**
 * Initializer that generates tensors initialized to a constant value.
 *
 * @param value float; the value of the generator tensors.
*/
IConstant::IConstant(float value) : Initializer("constant") {
    // Todo: Implement
    this->value = value;
}

void IConstant::apply(Tensor* params)
{
  params->fill_(value);
}
