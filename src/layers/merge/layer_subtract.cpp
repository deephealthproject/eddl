/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
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

int LSubtract::total_layers = 0;

LSubtract::LSubtract(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LSubtract layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::sameShape(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LSubtract layers with different tensor shape");
            }

    if(name.empty()) this->name = "subtract" + to_string(++total_layers);

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(parent[0]->output->shape, dev);  }

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LSubtract::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LSubtract::forward() {
    // TODO: Implement
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LSubtract::backward() {
    // TODO: Implement
    for (int i = 0; i < parent.size(); ++i){
        Tensor::inc(delta, parent[i]->delta);
    }
}

Layer *LSubtract::share(int c, int bs, vector<Layer *> p) {
    LSubtract *n = new LSubtract(p, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LSubtract::clone(int c, int bs, vector<Layer *> p, int todev) {
    LSubtract *n = new LSubtract(p,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}





///////////////
