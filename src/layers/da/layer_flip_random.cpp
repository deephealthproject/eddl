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

int LFlipRandom::total_layers = 0;

LFlipRandom::LFlipRandom(Layer *parent, int axis, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "flip_random" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->axis = axis;

    parent->addchild(this);
    addparent(parent);

}


void LFlipRandom::forward() {
    Tensor::flip_random(this->input, this->output, this->axis);
}

void LFlipRandom::backward() {

}


Layer *LFlipRandom::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LFlipRandom(p[0], this->axis, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LFlipRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LFlipRandom(p[0], this->axis, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LFlipRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
