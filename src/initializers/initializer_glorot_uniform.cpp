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
#include <cmath>

#include "initializer.h"

using namespace std;

/**
 * Glorot uniform initializer, also called Xavier uniform initializer.
 *
 * It draws samples from a uniform distribution within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out))
 * where fan_in is the number of input units in the weight tensor and fan_out is the number of output
 * units in the weight tensor.
 *
 * @param seed int; Used to seed the random generator.
*/
IGlorotUniform::IGlorotUniform(int seed) : Initializer("glorot_uniform") {
    // Todo: Implement
    this->seed = seed;
}
void IGlorotUniform::apply(Tensor* params) {
    if (params->ndim == 1)
        params->rand_signed_uniform(0.1f);
    else if (params->ndim == 2)
        params->rand_normal(0.0f, ::sqrtf(2.0f / (float)params->shape[0]));
    else
        params->rand_normal(0.0f, ::sqrtf(2.0f / ((float)params->size / params->shape[0])));

}
