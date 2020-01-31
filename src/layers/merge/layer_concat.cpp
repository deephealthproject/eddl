/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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

LConcat::LConcat(vector<Layer *> parent, string name, int dev,int mem) : MLayer(name, dev) {
    if(name.empty()) {this->name = "concat" + to_string(++total_layers); }

    // Perform layer checks
    if (parent.empty()) { msg("Error: LConcat layer with empty list"); }
    this->ndim = parent[0]->output->ndim;
    mem_level=mem;

    if (parent.size() > 1) {
        // All layers need to have the same number of dimensions
        for (int i = 0; i < parent.size() - 1; ++i) {
            if (ndim != parent[i]->output->ndim){
                msg("Error: LConcat layers with different tensor dims");
            }
        }

        // Check that all dimensions except depth (d=1), match
        if (ndim == 2) {
            for (int i = 0; i < parent.size() - 1; ++i){
                if (parent[i]->output->shape[0] != parent[i + 1]->output->shape[0]){
                    msg("Error: LConcat layers with different size in dim 1");
                }
            }
        } else if (ndim == 4) {
            for (int i = 0; i < parent.size() - 1; ++i) {
                if (parent[i]->output->shape[0] != parent[i + 1]->output->shape[0]){
                    msg("Error: LConcat layers with different size in dim 1, batch size");
                } else if (parent[i]->output->shape[2] != parent[i + 1]->output->shape[2]){
                    msg("Error: LConcat layers with different size in dim 3, rows of 4D");
                } else if (parent[i]->output->shape[3] != parent[i + 1]->output->shape[3]){
                    msg("Error: LConcat layers with different size in dim 4, cols of 4D");
                }
            }
        } else {
            msg("Error: LConcat layers of 2D or 4D tensors");
        }
    }

    // Sum depth dimensions to know the final dimension [2+2+6=10]
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->shape[1];
        index.push_back(t);
    }

    // Set new shape
    vector<int> shape = parent[0]->output->getShape();
    shape[1] = t;

    input = parent[0]->output;
    output = new Tensor(shape, dev);
    if (mem_level<2) delta = new Tensor(shape, dev);

    // Create a descriptor for each layer address translation
    int temp = 0;
    for (int i = 0; i < parent.size(); ++i) {
        auto *aux = new SelDescriptor({":", to_string(temp)+":"+to_string(temp+parent[i]->output->shape[1])});
        temp += parent[i]->output->shape[1];

        aux->build(this->output->shape);
        aux->build_indices();
        this->sd.push_back(aux);
    }

    // Set childs
    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual
void LConcat::forward() {
    // Copy all parent->output (tensor) to a section of this output (tensor)
    for (int i = 0; i < this->parent.size(); ++i) {
        // cout << this->name << endl;
        // this->parent[i]->output->info();
        Tensor::set_select(this->output, this->parent[i]->output, this->sd[i]);
    }
}


void LConcat::backward() {
    for (int i = 0; i < parent.size(); ++i) {
        if (parent[i]->mem_level==2) parent[i]->mem_delta();
        Tensor::set_select_back(this->delta, parent[i]->delta, this->sd[i]);
    }
    if (mem_level==2) free_delta();
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

    LConcat *n = new LConcat(p, "clone_" + to_string(todev) + name, todev,mem_level);
    n->orig = this;

    return n;
}


string LConcat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
