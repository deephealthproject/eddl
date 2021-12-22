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

#include "eddl/layers/operators/layer_operators.h"


using namespace std;

int LMult::total_layers = 0;

LMult::LMult(Layer *l1, Layer *l2, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "mult_" + to_string(++total_layers);
    binary = 1;

    input=l1->output;
    output = new Tensor(l1->output->shape, dev);

    l1->addchild(this);
    l2->addchild(this);
    addparent(l1);
    addparent(l2);
}

/**
  @brief Computes the sum operation between a layer and a float

  @param l a Layer.
  @param k a float.
  @param name a name for the operation (predefined as 'sum+TotaLMultLayers')
  @param dev which computing service utilize

  @returns the result of l+k element-wise over l

  */
LMult::LMult(Layer *l, float k, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "mult_" + to_string(++total_layers);
    val = k;

    input=l->output;
    output = new Tensor(l->output->shape, dev);


    l->addchild(this);
    addparent(l);
}

void LMult::forward() {
    if (binary) Tensor::el_mult(parent[0]->output, parent[1]->output, output, 0);
    else {
        Tensor::mult(parent[0]->output, output, val);
    }
}

void LMult::backward() {
    if (binary) {
        Tensor::el_mult(delta,parent[0]->output,parent[1]->delta,1);
        Tensor::el_mult(delta,parent[1]->output,parent[0]->delta,1);
    }
    else {
        delta->mult_(val);
        Tensor::inc(delta,parent[0]->delta);
    }
}

Layer *LMult::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LMult::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMult *n;
    if (binary)
        n = new LMult(p[0], p[1],  name, todev, this->mem_level);
    else
        n = new LMult(p[0], val,  name, todev, this->mem_level);
    n->orig = this;
    return n;
}
