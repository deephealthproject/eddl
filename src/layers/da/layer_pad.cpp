/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LPad::total_layers = 0;

LPad::LPad(Layer *parent, vector<int> pads, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "pad" + to_string(++total_layers);

    if(pads.size()!=2){
        msg("Pads must be a vector of int of size 2: (top-bottom, left-right)", "Layer:LPad");
    }
    input = parent->output;
    output = new Tensor({input->shape[0], input->shape[1], input->shape[2]+pads[0], input->shape[3]+pads[1]}, dev);

    // Params
    this->pads = pads;

    parent->addchild(this);
    addparent(parent);
}


void LPad::forward() {
    Tensor::pad(this->input, this->output, this->pads);
}

void LPad::backward(){
    Tensor::pad_back(this->input, this->output, this->pads);
}


Layer *LPad::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LPad(p[0], this->pads, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LPad::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LPad(p[0], this->pads,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LPad::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
