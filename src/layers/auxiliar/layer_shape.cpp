/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/auxiliar/layer_auxiliar.h"


using namespace std;

int LShape::total_layers = 0;

LShape::LShape(Layer *parent, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "shape" + to_string(++total_layers);

    input = parent->output;

    // Copy and cast parent's output shape
    vector<float> data(input->shape.begin(), input->shape.end());
    output = new Tensor(data, input->shape, dev);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LShape::resize(int batch){

}


void LShape::forward() {

}

void LShape::backward() {
    msg("NotImplementedError", "LShape::backward");
}


Layer *LShape::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LShape(p[0], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LShape::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LShape(p[0], name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LShape::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
