/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
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

int LRotateRandom::total_layers = 0;

LRotateRandom::LRotateRandom(Layer *parent, vector<float> factor, vector<int> offset_center, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "rotate_random" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    // Params
    this->factor = factor;
    this->offset_center = offset_center;
    this->da_mode = da_mode;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}

LRotateRandom::~LRotateRandom()
{
  delta=nullptr;
}

// virtual
void LRotateRandom::resize(int batch){
  output->resize(batch);
}

void LRotateRandom::forward() {
    Tensor::rotate_random(this->input, this->output, this->factor, this->offset_center, this->da_mode, this->constant);
}

void LRotateRandom::backward() {

}


Layer *LRotateRandom::share(int c, int bs, vector<Layer *> p) {
    LRotateRandom *n = new LRotateRandom(p[0], this->factor, this->offset_center, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LRotateRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRotateRandom *n = new LRotateRandom(p[0], this->factor, this->offset_center, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRotateRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
