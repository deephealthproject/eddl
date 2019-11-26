/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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

LCrop::LCrop(Layer *parent, vector<int> from_coords, vector<int> to_coords, bool reshape, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "crop" + to_string(++total_layers);

    input = parent->output;
    delta = parent->delta;

    if (reshape){
        output = new Tensor({input->shape[0], input->shape[1], to_coords[0]-from_coords[0]+1, to_coords[1]-from_coords[1]+1}, dev);
    }{
        output = new Tensor(input->getShape(), dev);
    }

    // Params
    this->from_coords = from_coords;
    this->to_coords = to_coords;
    this->reshape = reshape;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}
LCrop::~LCrop()
{
  delta=nullptr;
}

// virtual
void LCrop::resize(int batch){
  output->resize(batch);
}

void LCrop::forward() {
    Tensor::crop(this->input, this->output, this->from_coords, this->to_coords, this->constant);
}

void LCrop::backward(){

}


Layer *LCrop::share(int c, int bs, vector<Layer *> p) {
    LCrop *n = new LCrop(p[0], this->from_coords, this->to_coords, this->reshape, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LCrop::clone(int c, int bs, vector<Layer *> p, int todev) {
    LCrop *n = new LCrop(p[0], this->from_coords, this->to_coords, this->reshape, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LCrop::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
