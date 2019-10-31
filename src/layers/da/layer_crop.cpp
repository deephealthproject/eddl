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

int LCrop::total_layers = 0;

LCrop::LCrop(Layer *parent, bool reshape, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "crop" + to_string(++total_layers);

    // TODO: Implement
    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    //delta = parent->delta;

    this->reshape = reshape;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LCrop::resize(int batch){
  output->resize(batch);
}

void LCrop::forward() {
  this->output = Tensor::crop(this->input, {1,1,1,1}, {1,1,1,1}, this->reshape, this->constant);
}

void LCrop::backward() {

}


Layer *LCrop::share(int c, int bs, vector<Layer *> p) {
    LCrop *n = new LCrop(p[0], this->reshape, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LCrop::clone(int c, int bs, vector<Layer *> p, int todev) {
    LCrop *n = new LCrop(p[0], this->reshape, this->constant, "clone_" + to_string(todev) + name, todev);
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
