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

#include "metric.h"

using namespace std;


MCategoricalAccuracy::MCategoricalAccuracy() : Metric("categorical_accuracy"){}

float MCategoricalAccuracy::value(Tensor *T, Tensor *Y) {
    float f;
    f = accuracy(T, Y);
    return f;
}
