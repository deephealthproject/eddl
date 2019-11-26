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
#include <utility>

#include "layer_da.h"


using namespace std;

int LShift::total_layers = 0;

LShift::LShift(Layer *parent, vector<int> shift, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "shift" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    // Params
    this->shift = std::move(shift);
    this->da_mode = std::move(da_mode);
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}

LShift::~LShift()
{
  delta=nullptr;
}

// virtual
void LShift::resize(int batch){
  output->resize(batch);
}

void LShift::forward() {
    Tensor::shift(this->input, this->output, this->shift, this->da_mode, this->constant);
}

void LShift::backward() {

}


Layer *LShift::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LShift(p[0], this->shift, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LShift::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LShift(p[0], this->shift, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LShift::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
