/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/auxiliar/layer_auxiliar.h"


using namespace std;

int LTopK::total_layers = 0;

LTopK::LTopK(Layer *parent, vector<int> K_shape, string name, int dev, int mem, int axis, int largest, int sorted, int K) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "topk" + to_string(++total_layers);

    input = parent->output;

    output = new Tensor(K_shape, dev);

    this->axis = axis;
    this->largest = largest;
    this->sorted = sorted;
    this->K = K;

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LTopK::resize(int batch){
    output->resize(batch);
}


void LTopK::forward() {
    tensorNN::topK(this->input, this->output, axis, largest, sorted, K);
}

void LTopK::backward() {
    printf("Error, topk layer does not support backward\n"); exit(1);
}


Layer *LTopK::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LTopK(p[0], this->output->getShape(), "topk_"+to_string(c)+this->name, this->dev, this->mem_level, this->axis, this->largest, this->sorted, this->K);
    n->orig = this;

    return n;
}

Layer *LTopK::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LTopK(p[0], this->output->getShape(), name, todev, this->mem_level, this->axis, this->largest, this->sorted, this->K);
    n->orig = this;

    return n;
}


string LTopK::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
