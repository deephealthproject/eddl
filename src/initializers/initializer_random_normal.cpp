/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/initializers/initializer.h"

using namespace std;

/**
 * Initializer that generates tensors with a normal distribution.
 *
 * @param mean float; Mean of the random values to generate.
 * @param stdev float; Standard deviation of the random values to generate.
 * @param seed int; Used to seed the random generator
*/
IRandomNormal::IRandomNormal(float mean, float stdev, int seed) : Initializer("random_normal") {
    // Todo: Implement
    this->mean = mean;
    this->stdev = stdev;
    this->seed = seed;

}
void IRandomNormal::apply(Tensor* params)
{
  params->rand_normal(mean,stdev);
}
