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

int LQuantizeLinear::total_layers = 0;

LQuantizeLinear::LQuantizeLinear(Layer *parent, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "quantize_linear" + to_string(++total_layers);

    input = parent->output;

    output = new Tensor(input->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LQuantizeLinear::resize(int batch){
    output->resize(batch);
}


void LQuantizeLinear::forward() {
}

void LQuantizeLinear::backward() {
    printf("Error, quantize_linear layer does not support backward\n"); exit(1);
}


Layer *LQuantizeLinear::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LQuantizeLinear(p[0], "multithreshold_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LQuantizeLinear::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LQuantizeLinear(p[0], name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LQuantizeLinear::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
