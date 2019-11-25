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

int LRotateRandom::total_layers = 0;

LRotateRandom::LRotateRandom(Layer *parent, vector<float> factor, vector<int> offset_center, string da_mode, float constant, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "rotate_random" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = parent->delta;

    // Params
    this->factor = factor;
    this->offset_center = offset_center;
    this->da_mode = da_mode;
    this->constant = constant;

    parent->addchild(this);
    addparent(parent);

}

LRotateRandom::~LRotateRandom()
{
  delta=nullptr;
}

// virtual
void LRotateRandom::resize(int batch){
  output->resize(batch);
}

void LRotateRandom::forward() {
    auto *A=new Tensor({1, input->shape[1], input->shape[2], input->shape[3]}, input->device);
    int idx = (int)uniform(0.0f, (float)input->shape[0]-1.0f);
    A->ToGPU();
    Tensor::select(input, A, {idx}, 0, 1);
    A->ToCPU();
    A->save("images/test_da_" + to_string(idx) + "_0.jpg");

    // Method
    Tensor::rotate_random(this->input, this->output, this->factor, this->offset_center, this->da_mode, this->constant);

    auto *B=new Tensor({1, output->shape[1], output->shape[2], output->shape[3]}, output->device);
    B->ToGPU();
    Tensor::select(output, B, {idx}, 0, 1);
    B->ToCPU();
    B->save("images/test_da_" + to_string(idx) + "_1.jpg");
}

void LRotateRandom::backward() {

}


Layer *LRotateRandom::share(int c, int bs, vector<Layer *> p) {
    LRotateRandom *n = new LRotateRandom(p[0], this->factor, this->offset_center, this->da_mode, this->constant, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LRotateRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRotateRandom *n = new LRotateRandom(p[0], this->factor, this->offset_center, this->da_mode, this->constant, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRotateRandom::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
