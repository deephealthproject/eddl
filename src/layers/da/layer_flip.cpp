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

int LFlip::total_layers = 0;

LFlip::LFlip(Layer *parent, int axis, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "flip" + to_string(++total_layers);

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    //Params
    this->axis = axis;

    parent->addchild(this);
    addparent(parent);

}

LFlip::~LFlip()
{
  delta=nullptr;
}
// virtual
void LFlip::resize(int batch){
  output->resize(batch);
}

void LFlip::forward() {
    Tensor::flip(this->input, this->output, this->axis);
}

void LFlip::backward() {

}


Layer *LFlip::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LFlip(p[0], this->axis, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LFlip::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LFlip(p[0], this->axis, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LFlip::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
