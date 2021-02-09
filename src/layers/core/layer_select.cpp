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
LSelect::LSelect(Layer *parent, vector<string> indices, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    // Set default name
    if(name.empty()) this->name = "select_" + to_string(++total_layers);

    // Set input
    input=parent->output;

    // Build descriptor
    vector<int> shape_no_batch(input->shape.begin()+1, input->shape.end());
    sd = new SelDescriptor(indices, dev);
    sd->build(shape_no_batch);  // Ignore batch

    // Set flow tensors
    vector<int> oshape(sd->oshape);
    oshape.insert(oshape.begin() + 0, 1);
    output=new Tensor(oshape, dev);
//    delta=new Tensor(sd->oshape, dev);

    parent->addchild(this);
    addparent(parent);
}

LSelect::~LSelect(){
    delete sd;
}

void LSelect::resize(int b){
    Layer::resize(b);
    sd->resize(b);  // The batch is ignored
}

void LSelect::forward(){
    tensorNN::select(this->input, this->output, sd);
}

void LSelect::backward(){
    tensorNN::select_back(this->delta, this->parent[0]->delta, sd);
}


Layer *LSelect::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LSelect::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LSelect(p[0], sd->indices, name, todev, this->mem_level);
    n->orig = this;
    return n;
}
