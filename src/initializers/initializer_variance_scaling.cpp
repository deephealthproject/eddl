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
 * Initializer capable of adapting its scale to the shape of weights.
 *
 * @param scale float; Scaling factor. Positive float.
 * @param mode string; One of "fan_in", "fan_out", "fan_avg".
 * @param distribution string; Random distribution to use. one of "normal", "uniform".
 * @param seed int; Used to seed the random generator
*/
IVarianceScaling::IVarianceScaling(float scale, string mode, string distribution, int seed) : Initializer("variance_scaling") {
    // Todo: Implement
    this->scale = scale;
    this->mode = mode;
    this->distribution = distribution;
    this->scale = scale;

}
void IVarianceScaling::apply(Tensor* params)
{

	//TODO IMPLEMENT
	msg("Not Implemented", "Initializer::IVarianceScaling");
	
}
