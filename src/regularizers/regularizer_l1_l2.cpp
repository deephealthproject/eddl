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


RL1_L2::RL1_L2(float l1, float l2) : Regularizer("l1_l2") {
    // Todo: Implement
    this->l1 = l1; // regularization factor for l1
    this->l2 = l2; // regularization factor for l1
}

float RL1_L2::apply(Tensor* T) { return 0; }
