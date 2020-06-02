/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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

int LRotateRandom::total_layers = 0;

LRotateRandom::LRotateRandom(Layer *parent, vector<float> factor, vector<int> offset_center, WrappingMode da_mode, float cval, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "rotate_random" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->factor = factor;
    this->offset_center = offset_center;
    this->da_mode = da_mode;
    this->cval = cval;

    parent->addchild(this);
    addparent(parent);

}


void LRotateRandom::forward() {
    if (mode == TRMODE) {
        Tensor::rotate_random(this->input, this->output, this->factor, this->offset_center, this->da_mode, this->cval);
    } else {
        Tensor::copy(input, output);
    }
}

void LRotateRandom::backward() {

}


Layer *LRotateRandom::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LRotateRandom(p[0], this->factor, this->offset_center, this->da_mode, this->cval, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LRotateRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LRotateRandom(p[0], this->factor, this->offset_center, this->da_mode, this->cval,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LRotateRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
