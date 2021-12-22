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

#include "eddl/metrics/metric.h"

using namespace std;


MCategoricalAccuracy::MCategoricalAccuracy() : Metric("categorical_accuracy"){}

float MCategoricalAccuracy::value(Tensor *T, Tensor *Y) {
    float f;
    f = tensorNN::accuracy(T, Y);
    return f;
}

Metric* MCategoricalAccuracy::clone() {
    return new MCategoricalAccuracy();
}
