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

int LExpand::total_layers = 0;

LExpand::LExpand(Layer *parent, int size, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "expand" + to_string(++total_layers);

    this->size = size;
    input = parent->output;

    // Build descriptor
    auto *sd = new ExpandDescriptor(size, input->device);
    sd->build(input->shape);

    // Initialize tensor
    output = new Tensor(sd->oshape, input->device);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LExpand::resize(int batch){
    output->resize(batch);
}


void LExpand::forward() {
    Tensor::expand(this->input, this->output, this->sd);
}

void LExpand::backward() {
    msg("NotImplementedError", "LExpand::backward");
}


Layer *LExpand::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LExpand(p[0], this->size, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LExpand::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LExpand(p[0], this->size, name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LExpand::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
