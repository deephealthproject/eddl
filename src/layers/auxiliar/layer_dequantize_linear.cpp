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

int LDequantizeLinear::total_layers = 0;

LDequantizeLinear::LDequantizeLinear(Layer *parent, vector<int> shape, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "dequantize_linear" + to_string(++total_layers);
    this->shape = shape;

    if (parent != nullptr) {
      input = parent->output;
      output = new Tensor(input->shape, dev);
      parent->addchild(this);
      addparent(parent);
    } else {
      input = nullptr;
      output = new Tensor(shape, dev);
    }
}


// virtual
void LDequantizeLinear::resize(int batch){
    output->resize(batch);
}


void LDequantizeLinear::forward() {
}

void LDequantizeLinear::backward() {
    printf("Error, dequantize_linear layer does not support backward\n"); exit(1);
}


Layer *LDequantizeLinear::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LDequantizeLinear(p[0], this->shape, "multithreshold_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LDequantizeLinear::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LDequantizeLinear(p[0], shape, name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LDequantizeLinear::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
