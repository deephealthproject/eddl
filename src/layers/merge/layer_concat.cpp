/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
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

int LConcat::total_layers = 0;

LConcat::LConcat(vector<Layer *> parent, unsigned int axis, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if(name.empty()) {this->name = "concat" + to_string(++total_layers); }

    // Perform layer checks
    if (parent.empty()) { msg("Error: LConcat layer with empty list"); }
    this->ndim = parent[0]->output->ndim;
    this->axis = axis;

    if (parent.size() > 1) {

        // All layers need to have the same number of dimensions
        for (int i = 0; i < parent.size() - 1; ++i) {
            if (this->ndim != parent[i+1]->output->ndim){
                msg("Error: LConcat layers with different tensor dims");
            }
        }

        // Check dimensions
        for (int i = 0; i < parent.size() - 1; ++i){
            for (int d = 0; d < this->ndim; ++d) {
                if (d != this->axis && parent[i]->output->shape[d] != parent[i + 1]->output->shape[d]) {
                    msg("Error: LConcat layers with different size in dim 1");
                }
            }
        }

    }else{
        msg("Error: LConcat must receive at least two layers");
    }

    // Sum depth dimensions to know the final dimension [2+2+6=10]
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->shape[axis];
        index.push_back(t);
    }

    // Set new shape
    vector<int> new_shape = parent[0]->output->getShape();
    new_shape[axis] = t;

    input = parent[0]->output;
    output = new Tensor(new_shape, dev);
//    if (!mem_level) { delta = new Tensor(output->shape, dev);  }  // NOT parent[0]->output

    // Temp indices
    vector<string> indices_tmp;
    for (int i = 0; i < parent.size(); ++i) {
        indices_tmp.push_back(":");
    }

    // Create a descriptor for each layer address translation
    int start_idx = 0;
    int end_idx = 0;
    for (int i = 0; i < parent.size(); ++i) {
        // Build indices for the given axis
        start_idx = end_idx;
        end_idx = start_idx+parent[i]->output->shape[axis];
        vector<string> indices = vector<string>(indices_tmp);
        indices[axis] = to_string(start_idx)+":"+to_string(end_idx);

        // Build descriptor
        auto *aux = new SelDescriptor(indices);

        // Build translation table
        aux->build(this->output->shape);
        aux->build_indices();
        this->sd.push_back(aux);
    }

    // Set children
    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


void LConcat::forward() {
    // Get output tensors
    vector<Tensor*> outputs;
    for (auto & p : this->parent) { outputs.push_back(p->output); }

    // Perform concatenation
    Tensor::concat(outputs, this->axis, this->output);
}


void LConcat::backward() {
    // Get delta tensors
    vector<Tensor*> deltas;
    for (int i=0; i<this->parent.size(); i++) {
        // Store pointer of delta i
        deltas.push_back(parent[i]->delta);
    }

    // Perform concat (back)
    Tensor::concat_back(this->delta, deltas, this->axis);
}

Layer *LConcat::share(int c, int bs, vector<Layer *> p) {

    auto *n = new LConcat(p, this->axis, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LConcat::clone(int c, int bs, vector<Layer *> p, int todev) {

    auto *n = new LConcat(p, this->axis, name, todev,mem_level);
    n->orig = this;

    return n;
}


string LConcat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
