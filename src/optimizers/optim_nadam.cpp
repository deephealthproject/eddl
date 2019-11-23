/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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


Nadam::Nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay) : Optimizer() {
    this->lr = lr;
    this->beta_1 = beta_1;
    this->beta_2 = beta_2;
    this->epsilon = epsilon;
    this->schedule_decay = schedule_decay;

}
