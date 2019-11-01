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

int LShift::total_layers = 0;

LShift::LShift(Layer *parent, vector<float> factor, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "shift" + to_string(++total_layers);
    // factor => range of shift (0.8, 1.2)

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    //delta = parent->delta;

    // Params
    this->factor = factor;
    this->da_mode = da_mode;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LShift::resize(int batch){
  output->resize(batch);
}

void LShift::forward() {
    float rdn_factor = uniform(this->factor[0], this->factor[1]);
    vector<int> shift = {(int)(this->input->shape[2]*rdn_factor), (int)(this->input->shape[3]*rdn_factor)};
    //TODO: IMPLEMENT
//    this->output = Tensor::shift(this->input, shift, this->da_mode, this->constant);
}

void LShift::backward() {

}


Layer *LShift::share(int c, int bs, vector<Layer *> p) {
    LShift *n = new LShift(p[0], this->factor, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LShift::clone(int c, int bs, vector<Layer *> p, int todev) {
    LShift *n = new LShift(p[0], this->factor, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LShift::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
