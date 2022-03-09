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

int LWhere::total_layers = 0;

LWhere::LWhere(Layer *parent1, Layer *parent2, Layer *condition, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if(name.empty()) this->name = "where" + to_string(++total_layers);

    input = parent1->output; // Useless without a backward
    output = new Tensor(input->shape, dev);

    this->condition = condition->output;
    this->t_parent1 = parent1->output;
    this->t_parent2 = parent2->output;

    parent1->addchild(this);
    parent2->addchild(this);
    condition->addchild(this);

    addparent(parent1);
    addparent(parent2);
    addparent(condition);
}


// virtual
void LWhere::resize(int batch){
    output->resize(batch);
}


void LWhere::forward() {
    Tensor::where(this->condition, this->t_parent1, this->t_parent2, this->output);
}

void LWhere::backward() {
    Tensor::where_back(this->condition, parent[0]->delta, parent[1]->delta, this->delta);
}


Layer *LWhere::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LWhere(p[0], p[1], p[2], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LWhere::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LWhere(p[0], p[1], p[2], this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LWhere::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
