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

#include "regularizer.h"

using namespace std;


RL2::RL2(float value) : Regularizer("l2") {
    // Todo: Implement
    this->l = l; // regularization factor
}

float RL2::apply(Tensor* T) { return 0; }
