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

#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/generators/layer_generators.h"


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

LUniform::LUniform(float low, float high, vector<int> size, string name, int dev, int mem) : GeneratorLayer(name, dev, mem) {
    if(name.empty()) this->name = "generator_uniform" + to_string(++total_layers);

    this->low=low;
    this->high=high;

}

LUniform::~LUniform(){
    delete mask;
}


void LUniform::forward(){
}

void LUniform::backward(){
}

Layer *LUniform::share(int c, int bs, vector<Layer *> p) {
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LUniform::clone(int c, int bs, vector<Layer *> p, int todev) {
    LUniform *n;
    n = new LUniform(low, high, size, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
