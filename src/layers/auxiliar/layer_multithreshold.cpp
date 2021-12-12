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

int LMultiThreshold::total_layers = 0;

LMultiThreshold::LMultiThreshold(Layer *parent, vector<int> thresholds_shape, string name, int dev, int mem, float out_bias, float out_scale) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "multithreshold" + to_string(++total_layers);

    this->size = size;
    input = parent->output;

    thresholds = new Tensor(thresholds_shape, dev);
    this->out_bias = out_bias;
    this->out_scale = out_scale;

    output = new Tensor(input->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LMultiThreshold::resize(int batch){
    output->resize(batch);
}


void LMultiThreshold::forward() {
    tensorNN::multithreshold(this->input, this->output, this->thresholds, this->out_bias, this->out_scale);
}

void LMultiThreshold::backward() {
    printf("Error, multithreshold layer does not support backward\n"); exit(1);
}


Layer *LMultiThreshold::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LMultiThreshold(p[0], this->thresholds->getShape(), "multithreshold_"+to_string(c)+this->name, this->dev, this->mem_level, this->out_bias, this->out_scale);
    n->orig = this;

    return n;
}

Layer *LMultiThreshold::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LMultiThreshold(p[0], this->thresholds->getShape(), name, todev, this->mem_level, this->out_bias, this->out_scale);
    n->orig = this;

    return n;
}


string LMultiThreshold::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
