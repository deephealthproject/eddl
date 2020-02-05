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

int LScale::total_layers = 0;

LScale::LScale(Layer *parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "scale" + to_string(++total_layers);

    this->input = parent->output;
    delta=parent->delta;


    if (reshape){
        output = new Tensor({this->input->shape[0], this->input->shape[1], new_shape[0], new_shape[1]}, dev);
    }else{
        output = new Tensor(input->shape, dev);
    }

    // Params
    this->new_shape = new_shape;
    this->reshape = reshape;
    this->constant = constant;
    this->da_mode = da_mode;

    parent->addchild(this);
    addparent(parent);

}

LScale::~LScale()
{
  delta=nullptr;
}


// virtual


void LScale::forward() {
    Tensor::scale(this->input, this->output, this->new_shape, this->da_mode, this->constant);
}

void LScale::backward() {

}


Layer *LScale::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LScale(p[0], this->new_shape, this->reshape, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LScale::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LScale(p[0], this->new_shape, this->reshape, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LScale::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
