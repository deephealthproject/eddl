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

int LRotate::total_layers = 0;

LRotate::LRotate(Layer *parent, float angle, vector<int> offset_center, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "rotate" + to_string(++total_layers);

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta=parent->delta;


    // Params
    this->angle = angle;
    this->offset_center = offset_center;
    this->da_mode = da_mode;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}

LRotate::~LRotate()
{
  delta=nullptr;
}

// virtual


void LRotate::forward() {
    Tensor::rotate(this->input, this->output, angle, this->offset_center, this->da_mode, this->constant);
}

void LRotate::backward() {

}


Layer *LRotate::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LRotate(p[0], this->angle, this->offset_center, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LRotate::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LRotate(p[0], this->angle, this->offset_center, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LRotate::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
