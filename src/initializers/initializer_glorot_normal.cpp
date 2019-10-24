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
 * Glorot normal initializer, also called Xavier normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out))
 * where fan_in is the number of input units in the weight tensor and fan_out is the number of output units
 * in the weight tensor.
 *
 * @param seed int; Used to seed the random generator.
*/
IGlorotNormal::IGlorotNormal(int seed) : Initializer("glorot_normal") {
    // Todo: Implement
    this->seed = seed;
}
void IGlorotNormal::apply(Tensor* params){}
