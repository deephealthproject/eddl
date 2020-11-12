/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_RANDOM_H
#define EDDL_RANDOM_H

float gaussgen();
void build_randn_table();

float uniform(float min=0.0f, float max=1.0f);
float signed_uniform();

float slow_randn(float mean, float sd);
float fast_randn(float mean, float sd, int seed);


#endif //EDDL_RANDOM_H
