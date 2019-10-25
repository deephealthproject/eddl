/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "initializer.h"

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

void IConstant::apply(Tensor* params){}
