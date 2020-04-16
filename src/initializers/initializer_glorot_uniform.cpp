/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>

#include "eddl/initializers/initializer.h"

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
    else if (params->ndim == 2) {
        params->rand_signed_uniform(1.0);
        float limits=sqrtf(6.0f / (params->shape[0]+params->shape[1]));
        params->mult_(limits);
      }
    else if (params->ndim == 4) { // only fan_in
        params->rand_signed_uniform(1.0);
        float limits=sqrtf(3.0f / (params->shape[1]+params->shape[2]+params->shape[3]));
        params->mult_(limits);
    }
    else {
      params->rand_signed_uniform(0.1f);
    }

}
