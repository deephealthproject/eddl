/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/initializers/initializer.h"

using namespace std;

/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * @param minval float; Lower bound of the range of random values to generate.
 * @param maxval float; Upper bound of the range of random values to generate.
 * @param seed int; Used to seed the random generator.
*/
IRandomUniform::IRandomUniform(float minval, float maxval, int seed) : Initializer("random_uniform") {
    // Todo: Implement
    this->minval = minval;
    this->maxval = maxval;
    this->seed = seed;

}
void IRandomUniform::apply(Tensor* params)
{
    params->fill_rand_uniform_(1.0);
  params->mult_(maxval-minval);
  params->add_(minval);
}
