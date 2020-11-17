/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/optimizers/optim.h"

using namespace std;


Adamax::Adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay) : Optimizer() {
    this->lr = lr;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->weight_decay = weight_decay;

}


Adamax::~Adamax() {
    for(int i=0; i<mT.size(); i++){ delete mT[i]; }
}