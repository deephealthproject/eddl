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

int LPermute::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotaLPermuteLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LPermute::LPermute(Layer *parent, vector<int> dims, string name, int dev): OperatorLayer(name, dev) {
    // Set default name
    if(name.empty()) this->name = "permute_" + to_string(++total_layers);

    // Set input
    input=parent->output;

    // Temp input/output shape
    vector<int> ishape = vector<int>(input->shape.begin() + 1, input->shape.end());
    vector<int> oshape = permute_shape(ishape, dims);
    oshape.insert(oshape.begin(), 1);  // Insert batch (1, default), the it's resized

    // Compute index translation (output=>input)
    this->oi_addresses = permute_indices(ishape, dims);

    // Set flow tensors
    output=new Tensor(oshape, dev);
    delta=new Tensor(output->shape, dev);

    parent->addchild(this);
    addparent(parent);
}

void LPermute::forward(){
    Tensor::select(this->input, this->output, this->oi_addresses);
}

void LPermute::backward(){
    Tensor::select_back(this->delta, this->parent[0]->delta, this->oi_addresses);
}

void LPermute::resize(int b){
    Layer::resize(b);
}

Layer *LPermute::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LPermute::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LPermute(p[0], this->dims, "share_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
