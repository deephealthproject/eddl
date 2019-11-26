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

int LShiftRandom::total_layers = 0;

LShiftRandom::LShiftRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "shift_random" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    // Params
    this->factor_x = factor_x;
    this->factor_y = factor_y;
    this->da_mode = da_mode;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}

LShiftRandom::~LShiftRandom()
{
  delta=nullptr;
}

// virtual
void LShiftRandom::resize(int batch){
  output->resize(batch);
}

void LShiftRandom::forward() {
    auto *A=new Tensor({1, input->shape[1], input->shape[2], input->shape[3]}, input->device);
    int idx = (int)uniform(0.0f, (float)input->shape[0]-1.0f);
    A->toGPU();
    Tensor::select(input, A, {idx}, 0, 1);
    A->toCPU();
    A->save("images/test_da_" + to_string(idx) + "_0.jpg");

    // Method
    Tensor::shift_random(input, output, factor_x, factor_y);

    auto *B=new Tensor({1, output->shape[1], output->shape[2], output->shape[3]}, output->device);
    B->toGPU();
    Tensor::select(output, B, {idx}, 0, 1);
    B->toCPU();
    B->save("images/test_da_" + to_string(idx) + "_1.jpg");
}

void LShiftRandom::backward() {

}


Layer *LShiftRandom::share(int c, int bs, vector<Layer *> p) {
    LShiftRandom *n = new LShiftRandom(p[0], this->factor_x, this->factor_y, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LShiftRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LShiftRandom *n = new LShiftRandom(p[0], this->factor_x, this->factor_y, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LShiftRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
