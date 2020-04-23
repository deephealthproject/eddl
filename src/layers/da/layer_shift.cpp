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
#include <utility>

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LShift::total_layers = 0;

LShift::LShift(Layer *parent, vector<int> shift, string da_mode, float constant, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "shift" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->shift = std::move(shift);
    this->da_mode = std::move(da_mode);
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}



void LShift::forward() {
    Tensor::shift(this->input, this->output, this->shift, this->da_mode, this->constant);
}

void LShift::backward() {

}


Layer *LShift::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LShift(p[0], this->shift, this->da_mode, this->constant,  this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LShift::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LShift(p[0], this->shift, this->da_mode, this->constant,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LShift::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
