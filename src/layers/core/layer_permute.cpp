/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LPermute::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotaLPermuteLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LPermute::LPermute(Layer *parent, vector<int> dims, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "permute_" + to_string(++total_layers);

    // Set input
    input=parent->output;

    // Build descriptor
    vector<int> shape_no_batch(input->shape.begin()+1, input->shape.end());
    sd = new PermuteDescriptor(dims, dev);
    sd->build(shape_no_batch);

    // Set flow tensors
    vector<int> oshape(sd->oshape);
    oshape.insert(oshape.begin() + 0, 1);
    output=new Tensor(oshape, dev);

    parent->addchild(this);
    addparent(parent);
}

LPermute::~LPermute(){
    delete sd;
}

void LPermute::resize(int b){
    Layer::resize(b);
    sd->resize(b); // The batch is ignored
}

void LPermute::forward(){
    tensorNN::select(this->input, this->output, sd);
}

void LPermute::backward(){
    tensorNN::select_back(this->delta, this->parent[0]->delta, sd);
}

Layer *LPermute::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LPermute::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LPermute(p[0], sd->dims,  name, todev, this->mem_level);
    n->orig = this;
    return n;
}
