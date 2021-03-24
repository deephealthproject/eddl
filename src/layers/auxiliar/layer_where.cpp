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

int LWhere::total_layers = 0;

LWhere::LWhere(Layer *parent, Layer *condition, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "where" + to_string(++total_layers);

    input = parent->output;
    output = new Tensor(input->shape, dev);
    this->condition = condition->output;

    parent->addchild(this);
    condition->addchild(this);
    addparent(parent);
    addparent(condition);
}


// virtual
void LWhere::resize(int batch){
    output->resize(batch);
}


void LWhere::forward() {
    Tensor::where(this->condition, input, output);
}

void LWhere::backward() {
    msg("NotImplementedError", "LWhere::backward");
}


Layer *LWhere::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LWhere(p[0], p[1], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LWhere::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LWhere(p[0], p[1], name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LWhere::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
