/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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

    if (params->ndim == 1) {
        int fin=params->shape[0];
        int fout=params->shape[0];

        params->fill_rand_signed_uniform_(1.0);

        float limits=sqrtf(6.0f / (float)(fin+fout));

        params->mult_(limits);
      }
    else if (params->ndim == 2) {
        params->fill_rand_signed_uniform_(1.0);
        float limits=sqrtf(6.0f / (params->shape[0]+params->shape[1]));
        params->mult_(limits);
      }
    else if (params->ndim == 4) { // EDDL (output_depth, input_depth, kr,kc)

        int rf=params->shape[2]*params->shape[3];
        int fin=rf*params->shape[1];
        int fout=rf*params->shape[0];

        params->fill_rand_signed_uniform_(1.0);
        float limits=sqrtf(6.0f / (float)(fin+fout));

        params->mult_(limits);

    }
    else {
        params->fill_rand_signed_uniform_(0.1f);
    }

}
