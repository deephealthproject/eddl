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

LCropRandom::LCropRandom(Layer *parent, vector<int> new_shape, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "crop_random" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor({input->shape[0], input->shape[1], new_shape[0], new_shape[1]}, dev);
    delta = parent->delta;

    // Params
    this->new_shape = std::move(new_shape);

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
    auto *A=new Tensor({1, input->shape[1], input->shape[2], input->shape[3]}, input->device);
    int idx = (int)uniform(0.0f, (float)input->shape[0]-1.0f);
    A->ToGPU();
    Tensor::select(input, A, {idx}, 0, 1);
    A->ToCPU();
    A->save("images/test_f_" + to_string(idx) + "_0.jpg");

    // Method
    Tensor::crop_random(this->input, this->output);

    auto *B=new Tensor({1, output->shape[1], output->shape[2], output->shape[3]}, output->device);
    B->ToGPU();
    Tensor::select(output, B, {idx}, 0, 1);
    B->ToCPU();
    B->save("images/test_f_" + to_string(idx) + "_1.jpg");
}

void LCropRandom::backward(){

}


Layer *LCropRandom::share(int c, int bs, vector<Layer *> p) {
    LCropRandom *n = new LCropRandom(p[0], this->new_shape, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LCropRandom::clone(int c, int bs, vector<Layer *> p, int todev) {
    LCropRandom *n = new LCropRandom(p[0], this->new_shape, "clone_" + to_string(todev) + name, todev);
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
