/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "loss.h"

using namespace std;


LMin::LMin() : Loss("min"){}

void LMin::delta(Tensor *T, Tensor *Y, Tensor *D) {
    D->fill_(1);
}

float LMin::value(Tensor *T, Tensor *Y) {
    float sum;
    sum=Y->sum();
    return sum;
}
