/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
//
// Created by Salva Carrión on 20/09/2019.
//

#ifndef EDDL_RANDOM_H
#define EDDL_RANDOM_H

float gaussgen();
void build_randn_table();

float uniform(float min=0.0f, float max=1.0f);
float signed_uniform();

float slow_randn(float mean, float sd);
float fast_randn(float mean, float sd, int seed);


#endif //EDDL_RANDOM_H
