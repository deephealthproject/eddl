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

int LShiftRandom::total_layers = 0;

LShiftRandom::LShiftRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, WrappingMode da_mode, float cval, string name, int dev, int mem) : LDataAugmentation(parent, name, dev, mem) {
    if(name.empty()) this->name = "shift_random" + to_string(++total_layers);

    output = new Tensor(input->shape, dev);

    // Params
    this->factor_x = factor_x;
    this->factor_y = factor_y;
    this->da_mode = da_mode;
    this->cval = cval;

    parent->addchild(this);
    addparent(parent);

}


void LShiftRandom::forward() {
  if (mode == TRMODE) {
    Tensor::shift_random(input, output, factor_x, factor_y);
  } else {
    Tensor::copy(input, output);
  }
}

void LShiftRandom::backward() {

}


Layer *LShiftRandom::share(int c, int bs, vector<Layer *> p) {
    LShiftRandom *n = new LShiftRandom(p[0], this->factor_x, this->factor_y, this->da_mode, this->cval, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LShiftRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LShiftRandom *n = new LShiftRandom(p[0], this->factor_x, this->factor_y, this->da_mode, this->cval,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LShiftRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
