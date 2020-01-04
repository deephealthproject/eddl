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

#include "layer_da.h"


using namespace std;

int LCropRandom::total_layers = 0;

LCropRandom::LCropRandom(Layer *parent, vector<int> new_shape, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "crop_random" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor({input->shape[0], input->shape[1], new_shape[0], new_shape[1]}, dev);
    delta = parent->delta;

    // Params
    this->new_shape = std::move(new_shape);

    parent->addchild(this);
    addparent(parent);
}
LCropRandom::~LCropRandom()
{
  delta=nullptr;
}


// virtual
void LCropRandom::resize(int batch){
  output->resize(batch);
}

void LCropRandom::forward() {
    Tensor::crop_random(this->input, this->output);
}

void LCropRandom::backward(){

}


Layer *LCropRandom::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LCropRandom(p[0], this->new_shape, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LCropRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LCropRandom(p[0], this->new_shape, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LCropRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
