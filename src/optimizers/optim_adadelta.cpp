/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/optimizers/optim.h"

using namespace std;


AdaDelta::AdaDelta(float lr, float rho, float epsilon, float weight_decay) : Optimizer() {
    this->lr = lr;
    this->rho = rho;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;

}

AdaDelta::~AdaDelta() {
    for(int i=0; i<mT.size(); i++){ delete mT[i]; }
}