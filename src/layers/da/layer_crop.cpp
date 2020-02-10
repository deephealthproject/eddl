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

int LCrop::total_layers = 0;

LCrop::LCrop(Layer *parent, vector<int> from_coords, vector<int> to_coords, bool reshape, float constant, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "crop" + to_string(++total_layers);

    // Reshape if needed (TODO: builds output tensor twice... First parent, then here)
    if (reshape){
        output = new Tensor({input->shape[0], input->shape[1], to_coords[0]-from_coords[0]+1, to_coords[1]-from_coords[1]+1}, dev);
    }else{
        output = new Tensor(input->shape, dev);
    }

    // Params
    this->from_coords = from_coords;
    this->to_coords = to_coords;
    this->reshape = reshape;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}


void LCrop::forward() {
    Tensor::crop(this->input, this->output, this->from_coords, this->to_coords, this->constant);
}

void LCrop::backward(){

}


Layer *LCrop::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LCrop(p[0], this->from_coords, this->to_coords, this->reshape, this->constant, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LCrop::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LCrop(p[0], this->from_coords, this->to_coords, this->reshape, this->constant, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LCrop::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
