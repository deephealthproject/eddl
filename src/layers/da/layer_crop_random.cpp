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

int LCropRandom::total_layers = 0;

LCropRandom::LCropRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "crop_random" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    // Params
    this->factor_x = std::move(factor_x);
    this->factor_y = std::move(factor_y);
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);
}
LCropRandom::~LCropRandom()
{
  delta=nullptr;
}


// virtual
void LCropRandom::resize(int batch){
  output->resize(batch);
}

void LCropRandom::forward() {
    Tensor::crop_random(this->input, this->output, this->factor_x, this->factor_y, this->constant);
}

void LCropRandom::backward(){

}


Layer *LCropRandom::share(int c, int bs, vector<Layer *> p) {
    LCropRandom *n = new LCropRandom(p[0], this->factor_x, this->factor_y, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LCropRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LCropRandom *n = new LCropRandom(p[0], this->factor_x, this->factor_y, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LCropRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
