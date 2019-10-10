/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../operators/layer_operators.h"
#include "layer_generators.h"


using namespace std;

int LUniform::total_layers = 0;

/**
  @brief Draw samples from a uniform distribution

  @param low Lower boundary of the output interval. All values generated will be greater than or equal to low.
  @param high Upper boundary of the output interval. All values generated will be less than high.
  @param name a name for the operation (predefined as 'abs+TotalAbsLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */

LUniform::LUniform(float low, float high, vector<int> size, string name, int dev): GeneratorLayer(name, dev) {
    // TODO: Implement
    if(name.empty()) this->name = "generator_uniform" + to_string(++total_layers);

    this->low=low;
    this->high=high;

    ////////////

}

void LUniform::forward(){
    // TODO: Implement
}

void LUniform::backward(){
  // TODO: Implement
}

Layer *LUniform::share(int c, int bs, vector<Layer *> p) {
    // TODO: Implement
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LUniform::clone(int c, int bs, vector<Layer *> p, int todev) {
    // TODO: Implement
    LUniform *n;
    n = new LUniform(low, high, size, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
