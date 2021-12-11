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

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LRotate::total_layers = 0;

LRotate::LRotate(Layer *parent, float angle, vector<int> offset_center, WrappingMode da_mode, float cval, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "rotate" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->angle = angle;
    this->offset_center = offset_center;
    this->da_mode = da_mode;
    this->cval = cval;

    parent->addchild(this);
    addparent(parent);

}


void LRotate::forward() {
    Tensor::rotate(this->input, this->output, angle, this->offset_center, this->da_mode, this->cval);
}

void LRotate::backward() {

}


Layer *LRotate::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LRotate(p[0], this->angle, this->offset_center, this->da_mode, this->cval, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LRotate::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LRotate(p[0], this->angle, this->offset_center, this->da_mode, this->cval,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LRotate::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
