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

#include "eddl/initializers/initializer.h"

using namespace std;

/**
 * Initializer that generates a random orthogonal matrix.
 *
 * @param gain float; Multiplicative factor to apply to the orthogonal matrix.
 * @param seed int; Used to seed the random generator.
*/
IOrthogonal::IOrthogonal(float gain, int seed) : Initializer("orthogonal") {
    // Todo: Implement
    this->gain = gain;
    this->seed = seed;

}
void IOrthogonal::apply(Tensor* params)
{
  msg("Orthogonalnot implemented", "IOrthogonal:apply");
}
