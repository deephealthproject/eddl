/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "initializers/initializer.h"

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
void IGlorotNormal::apply(Tensor* params)
{
  if (params->ndim == 1)
      params->rand_signed_uniform(0.1f);
  else if (params->ndim == 2)
      params->rand_normal(0.0f, ::sqrtf(2.0f / (params->shape[0]+params->shape[1])));
  else if (params->ndim == 4) // only fan_in
      params->rand_normal(0.0f, ::sqrtf(1.0f / ((float)params->size / params->shape[0])));
  else {
      params->rand_signed_uniform(0.1f);
  }
}
