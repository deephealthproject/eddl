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

int LEqual::total_layers = 0;

LEqual::LEqual(Layer *parent1, Layer *parent2, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "equal" + to_string(++total_layers);

    this->A = parent1->output;
    this->B = parent2->output;
    output = new Tensor(input->shape, dev);

    parent1->addchild(this);
    parent2->addchild(this);
    addparent(parent1);
    addparent(parent2);
}


// virtual
void LEqual::resize(int batch){
    output->resize(batch);
}


void LEqual::forward() {
    Tensor::equal(A, B, output);
}

void LEqual::backward() {
    msg("NotImplementedError", "LEqual::backward");
}


Layer *LEqual::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LEqual(p[0], p[1], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LEqual::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LEqual(p[0], p[1], name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LEqual::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
