/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/initializers/initializer.h"

using namespace std;

/**
 * He normal initializer
 *
 * It draws samples from a truncated normal distribution centered on 0 with stddev = sqrt(2 / (fan_in))
 * where fan_in is the number of input units in the weight tensor.
 *
 * @param seed int; Used to seed the random generator.
*/
IHeNormal::IHeNormal(int seed) : Initializer("He_normal") {
    // Todo: Implement
    this->seed = seed;
}
void IHeNormal::apply(Tensor* params)
{
  if (params->ndim == 1)
      params->fill_rand_signed_uniform_(0.1f);
  else if (params->ndim == 2) {
      params->fill_rand_normal_(0.0f, ::sqrtf(2.0f / (params->shape[0])));
    }
  else if (params->ndim == 4) {
      int rf=params->shape[2]*params->shape[3];
      int fin=rf*params->shape[1];
      params->fill_rand_normal_(0.0f, ::sqrtf(2.0f / (float) (fin)));
    }
  else {
      params->fill_rand_signed_uniform_(0.1f);
  }
}
