/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/auxiliar/layer_auxiliar.h"


using namespace std;

int LShape::total_layers = 0;

LShape::LShape(Layer *parent, bool include_batch, string name, int dev, int mem) : LinLayer(name, dev, mem, "shape") {
    if(name.empty()) this->name = "shape" + to_string(++total_layers);

    // Set input
    input = parent->output;

    // Set vars
    this->include_batch = include_batch;

    // [DATA]: Copy data, but ignore batch optionally
    int offset = int(!include_batch);
    this->data = vector<float>(input->shape.begin() + offset, input->shape.end());

    // [Output]: Data=data, Shape=(1batch, data length)
    vector<int> oshape; oshape.push_back(1); oshape.push_back(data.size());
    output = new Tensor(this->data, oshape, this->dev);  // Must include batch

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LShape::resize(int batch){
    delete this->output;

    // Create tensor with batch 1
    vector<int> oshape; oshape.push_back(1); oshape.push_back(data.size());
    auto tmp = new Tensor(this->data, oshape, this->dev);  // batch 1

    // Create output tensor
    this->output = Tensor::repeat(tmp, batch, 0);
    delete tmp;
}


void LShape::forward() {
}

void LShape::backward() {
}


Layer *LShape::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LShape(p[0], this->include_batch, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LShape::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LShape(p[0], this->include_batch, name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LShape::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
