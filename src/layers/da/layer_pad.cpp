/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "eddl/layers/da/layer_da.h"


using namespace std;

int LPad::total_layers = 0;

LPad::LPad(Layer *parent, vector<int> padding, float constant, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "pad" + to_string(++total_layers);

    // Check params
    if(padding.size()==2){
        this->padding = {padding[0], padding[1], padding[0], padding[1]};
    } else if(padding.size()==4){ this->padding = padding; }
    else{
        msg("The padding on each border must follow this format (top-bottom, left-right) or (top, right, bottom, left)", "LPad::LPad");
    }
    this->constant = constant;

    input = parent->output;
    output = Tensor::full({input->shape[0], input->shape[1], input->shape[2]+(this->padding[0]+this->padding[2]), input->shape[3]+(this->padding[1]+this->padding[3])}, this->constant, dev);

    parent->addchild(this);
    addparent(parent);
}


void LPad::forward() {
    Tensor::pad(this->input, this->output, this->padding);
}

void LPad::backward(){
    Tensor::pad_back(parent[0]->delta, this->delta, this->padding);
}

void LPad::resize(int batch) {
    Layer::resize(batch);
    // We need to fill again the output tensor because the values are deleted in the resize function
    this->output->fill_(this->constant);
}

Layer *LPad::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LPad(p[0], this->padding, this->constant, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LPad::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LPad(p[0], this->padding, this->constant, name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LPad::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
