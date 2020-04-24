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

int LFlip::total_layers = 0;

LFlip::LFlip(Layer *parent, int axis, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "flip" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    //Params
    this->axis = axis;

    parent->addchild(this);
    addparent(parent);

}


void LFlip::forward() {
    Tensor::flip(this->input, this->output, this->axis);
}

void LFlip::backward() {

}


Layer *LFlip::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LFlip(p[0], this->axis, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LFlip::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LFlip(p[0], this->axis,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LFlip::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
