/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LCropScaleRandom::total_layers = 0;

LCropScaleRandom::LCropScaleRandom(Layer *parent, vector<float> factor, WrappingMode da_mode, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "crop_scale" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->factor=std::move(factor);
    this->da_mode=da_mode;

    parent->addchild(this);
    addparent(parent);
}


void LCropScaleRandom::forward() {
  if (mode == TRMODE) {
    Tensor::crop_scale_random(this->input, this->output, this->factor, this->da_mode);
  } else {
    Tensor::copy(input, output);
  }
}

void LCropScaleRandom::backward() {

}


Layer *LCropScaleRandom::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LCropScaleRandom(p[0], this->factor, this->da_mode, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LCropScaleRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LCropScaleRandom(p[0], this->factor, this->da_mode,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LCropScaleRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
