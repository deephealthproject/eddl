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

int LScaleRandom::total_layers = 0;

LScaleRandom::LScaleRandom(Layer *parent, vector<float> factor, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "scale_random" + to_string(++total_layers);

    this->input = parent->output;
    this->output = new Tensor(input->getShape(), dev);
    this->delta = parent->delta;

    // Params
    this->factor = std::move(factor);
    this->constant = constant;
    this->da_mode = da_mode;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LScaleRandom::resize(int batch){
  output->resize(batch);
}

void LScaleRandom::forward() {
    Tensor::scale_random(this->input, this->output, this->factor, this->da_mode, this->constant);
}

void LScaleRandom::backward() {

}


Layer *LScaleRandom::share(int c, int bs, vector<Layer *> p) {
    LScaleRandom *n = new LScaleRandom(p[0], this->factor, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LScaleRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LScaleRandom *n = new LScaleRandom(p[0], this->factor, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LScaleRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
