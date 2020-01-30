/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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
