/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <utility>

#include "layer_operators.h"


using namespace std;

int LSelect::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotaLSelectLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LSelect::LSelect(Layer *parent, vector<string> str_indices, string name, int dev): OperatorLayer(name, dev) {
    // Set default name
    if(name.empty()) this->name = "select_" + to_string(++total_layers);

    // Set input
    input=parent->output;

    // Get input shape and ranges of indices
    this->str_indices = std::move(str_indices);
    this->idxs_range = parse_indices(this->str_indices, vector<int>(input->shape.begin() + 1, input->shape.end()));

    // Get output shape (without batch)
    vector<int> oshape = indices2shape(idxs_range);
    oshape.insert(oshape.begin(), 1);  // Insert batch (1, default), the it's resized

    // Set flow tensors
    output=new Tensor(oshape, dev);
    delta=new Tensor(output->shape, dev);

    // Compute index translation (output=>input)
    this->oi_addresses = this->input->ranges2indices(this->idxs_range);

    parent->addchild(this);
    addparent(parent);
}

void LSelect::forward(){
    Tensor::select(this->input, this->output, this->oi_addresses);
}

void LSelect::backward(){
    Tensor::select_back(this->delta, this->parent[0]->delta, this->oi_addresses);
}

void LSelect::resize(int b){
    Layer::resize(b);
}

Layer *LSelect::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LSelect::clone(int c, int bs, vector<Layer *> p, int todev) {
    LSelect *n = new LSelect(p[0], this->str_indices, "share_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
