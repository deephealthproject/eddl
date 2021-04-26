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

#include "eddl/layers/auxiliar/layer_auxiliar.h"


using namespace std;

int LConstOfTensor::total_layers = 0;

LConstOfTensor::LConstOfTensor(Tensor *const_tensor, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "const_of_tensor" + to_string(++total_layers);

    this->const_tensor = const_tensor->clone(); this->const_tensor->toDevice(dev);
    input = output = Tensor::empty_like(this->const_tensor)->unsqueeze(0);   // Reserve memory and batch
}

LConstOfTensor::~LConstOfTensor(){
    if (output!=nullptr) { delete output; output=nullptr; }
    if (const_tensor!=nullptr) { delete const_tensor; const_tensor=nullptr; }
}


void LConstOfTensor::free_delta(){

    // DO NOT DELETE DELTA
    // There will be problems with network concatenation
    // [Input1]->[Net1]=>[Input2]->[Net2]->[Cost. func]
    // "=>" is a copyTensor(delta2, delta1)
    // If delta2 is deleted after the backward of Input2, there will be nothing to copy

    delta->fill_(0.0);
}

void LConstOfTensor::forward() {
    tensorNN::repeat_batch(this->const_tensor, output);
}

void LConstOfTensor::backward() {
}


Layer *LConstOfTensor::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LConstOfTensor(this->const_tensor, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LConstOfTensor::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LConstOfTensor(this->const_tensor, "share_"+to_string(c)+this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LConstOfTensor::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
