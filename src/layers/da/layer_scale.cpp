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

int LScale::total_layers = 0;

LScale::LScale(Layer *parent, vector<float> factor, bool reshape, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "scale" + to_string(++total_layers);
    // factor => range of scaling (0.8, 1.2)

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    //delta = parent->delta;

    // Params
    this->factor = factor;
    this->reshape = reshape;
    this->da_mode = da_mode;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LScale::resize(int batch){
  output->resize(batch);
}

void LScale::forward() {
    float rdn_factor = uniform(this->factor[0], this->factor[1]);
    vector<int> new_shape = {this->input->shape[2]*rdn_factor, this->input->shape[3]*rdn_factor};
    this->output = Tensor::scale(this->input, new_shape, this->reshape, this->da_mode, this->constant);
}

void LScale::backward() {

}


Layer *LScale::share(int c, int bs, vector<Layer *> p) {
    LScale *n = new LScale(p[0], this->factor, this->reshape, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LScale::clone(int c, int bs, vector<Layer *> p, int todev) {
    LScale *n = new LScale(p[0], this->factor, this->reshape, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LScale::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
