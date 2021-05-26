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

#include "eddl/layers/merge/layer_merge.h"


using namespace std;

int LMaximum::total_layers = 0;

LMaximum::LMaximum(vector<Layer *> parent, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if (parent.size() == 0) msg("Error: LMaximum layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::sameShape(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LMaximum layers with different tensor shape");
            }

    if(name.empty()) this->name = "maximum" + to_string(++total_layers);

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->shape, dev);


    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LMaximum::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LMaximum::forward() {
    // TODO: Implement
    output->fill_(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LMaximum::backward() {
    // TODO: Implement
    for (int i = 0; i < parent.size(); ++i){
        Tensor::inc(delta, parent[i]->delta);
    }
}

Layer *LMaximum::share(int c, int bs, vector<Layer *> p) {
    LMaximum *n = new LMaximum(p, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}


Layer *LMaximum::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMaximum *n = new LMaximum(p,  name, todev, this->mem_level);
    n->orig = this;

    return n;
}
