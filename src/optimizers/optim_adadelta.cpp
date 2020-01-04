/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "optim.h"

using namespace std;


AdaDelta::AdaDelta(float lr, float rho, float epsilon, float weight_decay) : Optimizer() {
    this->lr = lr;
    this->rho = rho;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;

}
