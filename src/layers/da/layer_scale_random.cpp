/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LScaleRandom::total_layers = 0;

LScaleRandom::LScaleRandom(Layer *parent, vector<float> factor, string da_mode, float constant, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "scale_random" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->factor = std::move(factor);
    this->constant = constant;
    this->da_mode = da_mode;

    parent->addchild(this);
    addparent(parent);

}



void LScaleRandom::forward() {
  if (mode == TRMODE) {
    Tensor::scale_random(this->input, this->output, this->factor, this->da_mode, this->constant);
  } else {
    Tensor::copy(input, output);
  }
}

void LScaleRandom::backward() {

}


Layer *LScaleRandom::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LScaleRandom(p[0], this->factor, this->da_mode, this->constant, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LScaleRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LScaleRandom(p[0], this->factor, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LScaleRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
