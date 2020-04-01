/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/merge/layer_merge.h"


using namespace std;

int LMatMul::total_layers = 0;

LMatMul::LMatMul(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LMatMul layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::eqsize(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LMatMul layers with different tensor shape");
            }

    if(name.empty()) this->name = "matmul" + to_string(++total_layers);

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(parent[0]->output->shape, dev);  }

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LMatMul::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LMatMul::forward() {
    // TODO: Implement
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LMatMul::backward() {
    // TODO: Implement
    for (int i = 0; i < parent.size(); ++i){
        Tensor::inc(delta, parent[i]->delta);
    }
}

Layer *LMatMul::share(int c, int bs, vector<Layer *> p) {
    LMatMul *n = new LMatMul(p, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LMatMul::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMatMul *n = new LMatMul(p, "share_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;

    return n;
}

