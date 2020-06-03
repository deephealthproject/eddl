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
#include <utility>

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LSelect::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotaLSelectLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LSelect::LSelect(Layer *parent, vector<string> indices, bool hasBatch, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "select_" + to_string(++total_layers);

    // Set input
    input=parent->output;

    // Add batch to indices (if needed)
    vector<string> indices_batch;
    if(hasBatch){
        indices_batch = indices;
    }else{
        indices_batch = vector<string>(indices);
        indices_batch.insert(indices_batch.begin(), ":");
    }

    // Build descriptor
    sd = new SelDescriptor(indices_batch);
    sd->build(input->shape);

    // Set flow tensors
    output=new Tensor(sd->oshape, dev);
//    delta=new Tensor(sd->oshape, dev);

    parent->addchild(this);
    addparent(parent);
}

void LSelect::resize(int b){
    Layer::resize(b);
    sd->resize(b);
}

void LSelect::forward(){
    Tensor::select(this->input, this->output, sd);
}

void LSelect::backward(){
    Tensor::select_back(this->delta, this->parent[0]->delta, sd);
}


Layer *LSelect::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LSelect::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LSelect(p[0], sd->indices, true,  name, todev, this->mem_level);
    n->orig = this;
    return n;
}
