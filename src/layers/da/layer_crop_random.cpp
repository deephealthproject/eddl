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

int LCropRandom::total_layers = 0;

LCropRandom::LCropRandom(Layer *parent, vector<int> new_shape, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "crop_random" + to_string(++total_layers);

    output = new Tensor({input->shape[0], input->shape[1], new_shape[0], new_shape[1]}, dev);

    // Params
    this->new_shape = std::move(new_shape);

    parent->addchild(this);
    addparent(parent);
}


void LCropRandom::forward() {
  if (mode == TRMODE) {
      Tensor::crop_random(this->input, this->output);
  } else {
      Tensor::copy(input, output);
  }

}

void LCropRandom::backward(){

}


Layer *LCropRandom::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LCropRandom(p[0], this->new_shape, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LCropRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LCropRandom(p[0], this->new_shape,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LCropRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
