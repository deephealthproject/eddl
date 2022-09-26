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

#include "eddl/layers/merge/layer_merge.h"
#include "eddl/layers/operators/layer_operators.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/profiling.h"
#include "eddl/hardware/fpga/fpga_hw.h"

using namespace std;

int LDConcat::total_layers = 0;

LDConcat::LDConcat(vector<Layer *> parent, unsigned int axis, string name, int dev, int mem) : MLayer(name, dev, mem) {
    if(name.empty()) {this->name = "dconcat" + to_string(++total_layers); }

    // Perform layer checks
    if (parent.empty()) { msg("Error: LDConcat layer with empty list"); }

    // Check special cases
    if(axis==-1){
        // -1 because last index is shape-1, and -1 (again) to remove the batch component
        axis = (parent[0]->output->ndim - 1) - 1;
    }else if(axis<0){
        msg("The axist must be greater or equal than zero", "LConcat::LConcat");
    }

    this->axis = axis; // batch is not included

    // Sum depth dimensions to know the final dimension [2+2+6=10]
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->shape[this->axis+1];
    }

    // Set new shape
    vector<int> new_shape = parent[0]->output->getShape();
    new_shape[this->axis+1] = t;

    input = parent[0]->output;
    output = new Tensor(new_shape, dev);

    // Set children
    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }
}

void LDConcat::forward() {}

void LDConcat::backward() {
}

Layer *LDConcat::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LDConcat(p, this->axis, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    return n;
}

Layer *LDConcat::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LDConcat(p, this->axis, name, todev,mem_level);
    n->orig = this;

    return n;
}


string LDConcat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
