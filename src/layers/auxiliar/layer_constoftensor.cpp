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

    this->const_tensor = const_tensor->clone();
    output = input = this->const_tensor;
}


// virtual
void LConstOfTensor::resize(int batch){
}


void LConstOfTensor::forward() {
}

void LConstOfTensor::backward() {
    msg("NotImplementedError", "LConstOfTensor::backward");
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
