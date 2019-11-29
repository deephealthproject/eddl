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
LSelect::LSelect(Layer *l, vector<vector<int>> indices, string name, int dev): OperatorLayer(name, dev) {
    // Set default name
    if(name.empty()) this->name = "select_" + to_string(++total_layers);

    input=l->output;

    vector<int> output_shape;
    for(int i=0; i<indices.size(); i++){
        output_shape.push_back(indices[i][1] - indices[i][0] + 1);
    }
    output = new Tensor(output_shape, dev);
    delta=new Tensor(l->output->getShape(), dev);

    this->indices = indices;

    l->addchild(this);
    addparent(l);
}

void LSelect::forward(){
    Tensor::select(this->input, this->output, this->indices);
}

void LSelect::backward(){
}

void LSelect::resize(int b){
  Layer::resize(b);
}

Layer *LSelect::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LSelect::clone(int c, int bs, vector<Layer *> p, int todev) {
    LSelect *n = new LSelect(p[0], this->indices, "share_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
