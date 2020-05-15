/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/merge/layer_merge.h"


using namespace std;

int LMinimum::total_layers = 0;

LMinimum::LMinimum(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LMinimum layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::eqsize(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LMinimum layers with different tensor shape");
            }

    if(name.empty()) this->name = "minimum" + to_string(++total_layers);

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(parent[0]->output->shape, dev);  }

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LMinimum::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LMinimum::forward() {
    // TODO: Implement
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LMinimum::backward() {
    // TODO: Implement
    for (int i = 0; i < parent.size(); ++i){
        Tensor::inc(delta, parent[i]->delta);
    }
}

Layer *LMinimum::share(int c, int bs, vector<Layer *> p) {
    LMinimum *n = new LMinimum(p, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LMinimum::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMinimum *n = new LMinimum(p,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}





///////////////
