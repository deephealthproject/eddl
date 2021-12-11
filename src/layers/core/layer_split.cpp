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
#include <utility>

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LSplit::total_layers = 0;


LSplit::LSplit(Layer *parent, vector<int> indexes, int axis, bool merge_sublayers, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "split_" + to_string(++total_layers);

    this->indexes = vector<int>(indexes);
    this->axis = axis;
    this->merge_sublayers = merge_sublayers;

    // Normalize axis
    if(axis==-1){
        axis = parent->output->ndim - 1;
    }else if(axis>=0){
        axis += 1;
    }

    // Check axis
    if(axis<0){
        msg("Axis must be equal or greater than zero, with the exception of '-1' to select the last axis. (Batch dim is ignored)", "layer::Split");
    }else if(axis>parent->output->ndim-1){
        msg("Axis must be smaller than the number of dimensions. (Batch dim is ignored)", "layer::Split");
    }

    // Check indexes values
    if(indexes.empty()) {
        msg("The indexes param can't be empty", "layer::Split");
    }else if(indexes.size() > parent->output->shape[axis]) {
        msg("There can't be more indexes than values", "layer::Split");
    }
    for(auto &idx : indexes){
        if(idx <= 0){
            msg("Indexes can't be zero or less", "layer::Split");
        }else if(idx > parent->output->shape[axis]){
            msg("Indexes greater than dimension size", "layer::Split");
        }
    }

    // Build ranges
    // g.g.: axis=2 => [{":", ":", "0:40", ":"}, {":", ":", "40:60", ":"},...]
    string last_idx = "0";
    for(int i=0; i<=indexes.size(); i++){ // [{}, {} ,...]; extra dim to "end"

        vector<string> sel_rngs;
        for(int j=1; j<parent->output->ndim;j++){ // {}; 0=batch, ignore
            if(j==axis){
                if (i==indexes.size()){ // last selection, to close
                    sel_rngs.push_back(last_idx + ":");
                }else{  // From 0 to n-2
                    sel_rngs.push_back(last_idx + ":" + to_string(indexes[i]));
                    last_idx = to_string(indexes[i]);
                }
            }else{
                sel_rngs.emplace_back(":");
            }
        }

        // Set sublayer name
        string lname = "split_"+ to_string(total_layers);
        if(!merge_sublayers){
            lname +="_"+ to_string(i+1);
        }

        // Add layers
        split_layers.push_back(new LSelect(parent, sel_rngs, lname, DEV_CPU, 0));
    }
}

LSplit::~LSplit(){
}

void LSplit::resize(int b){
}

void LSplit::forward(){
}

void LSplit::backward(){
}


Layer *LSplit::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LSplit::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LSplit(p[0], this->indexes, this->axis, this->merge_sublayers, name, todev, this->mem_level);
    n->orig = this;
    return n;
}
