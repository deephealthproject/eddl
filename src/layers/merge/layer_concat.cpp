/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_merge.h"


using namespace std;

int LConcat::total_layers = 0;

LConcat::LConcat(vector<Layer *> parent, string name, int dev) : MLayer(name, dev) {
    if (parent.size() == 0) msg("Error: LConcat layer with empty list");

    ndim = parent[0]->output->ndim;

    if (parent.size() > 1) {
        for (int i = 0; i < parent.size() - 1; ++i)
            if (ndim != parent[i]->output->ndim)
                msg("Error: LConcat layers with different tensor dims");

        if (ndim == 2) {
            for (int i = 0; i < parent.size() - 1; ++i)
                if (parent[i]->output->shape[0] != parent[i + 1]->output->shape[0])
                    msg("Error: LConcat layers with different size in dim 1");
        } else if (ndim == 4) {
            for (int i = 0; i < parent.size() - 1; ++i) {
                if (parent[i]->output->shape[0] != parent[i + 1]->output->shape[0])
                    msg("Error: LConcat layers with different size in dim 1, batch size");
                else if (parent[i]->output->shape[2] != parent[i + 1]->output->shape[2])
                    msg("Error: LConcat layers with different size in dim 3, rows of 4D");
                else if (parent[i]->output->shape[3] != parent[i + 1]->output->shape[3])
                    msg("Error: LConcat layers with different size in dim 4, cols of 4D");
            }
        } else {
            msg("Error: LConcat layers of 2D or 4D tensors");
        }
    }

    if(name.empty()) this->name = "concat" + to_string(++total_layers);

    input = parent[0]->output;
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->shape[1];
        index.push_back(t);
    }

    vector<int> shape = parent[0]->output->getShape();
    shape[1] = t;

    output = new Tensor(shape, dev);
    delta = new Tensor(shape, dev);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual
void LConcat::forward() {
    int ini = 0;
    for (int i = 0; i < parent.size(); ++i) {
        Tensor::fill(parent[i]->output, 0, parent[i]->output->shape[1], output, ini, index[i], 0);
        ini = index[i];
    }
}


void LConcat::backward() {

    if (parent.size()) {
        int ini = 0;
        for (int i = 0; i < parent.size(); ++i) {
            Tensor::fill(delta, ini, index[i], parent[i]->delta, 0, parent[i]->output->shape[1], 1);
            ini = index[i];
        }
    }
}

void LConcat::resize(int batch){
  Layer::resize(batch);
}

Layer *LConcat::share(int c, int bs, vector<Layer *> p) {

    LConcat *n = new LConcat(p, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LConcat::clone(int c, int bs, vector<Layer *> p, int todev) {

    LConcat *n = new LConcat(p, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LConcat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
