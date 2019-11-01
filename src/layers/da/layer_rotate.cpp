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

int LRotate::total_layers = 0;

LRotate::LRotate(Layer *parent, vector<float> factor, vector<int> axis, bool reshape, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "rotate" + to_string(++total_layers);

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    // Params
    this->factor = factor;
    this->axis = axis;
    this->reshape = reshape;
    this->da_mode = da_mode;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LRotate::resize(int batch){
  output->resize(batch);
}

void LRotate::forward() {
    // TODO: NOT IMPLEMENTED
    float rdn_angle = uniform(this->factor[0], this->factor[1]);
    Tensor::rotate(this->input, this->output, rdn_angle, {2,3}, this->da_mode, this->constant);
}

void LRotate::backward() {

}


Layer *LRotate::share(int c, int bs, vector<Layer *> p) {
    LRotate *n = new LRotate(p[0], this->factor, this->axis, this->reshape, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LRotate::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRotate *n = new LRotate(p[0], this->factor, this->axis, this->reshape, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRotate::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
