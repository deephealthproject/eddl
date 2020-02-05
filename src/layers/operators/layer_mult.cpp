/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
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

int LMult::total_layers = 0;

LMult::LMult(Layer *l1, Layer *l2, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "mult_" + to_string(++total_layers);
    binary = 1;

    input=l1->output;

    output = new Tensor(l1->output->shape, dev);
    if (!mem_level) { delta = new Tensor(l1->output->shape, dev);  }

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
    if (!mem_level) { delta = new Tensor(l->output->shape, dev);  }

    l->addchild(this);
    addparent(l);
}

void LMult::forward() {
    if (binary) Tensor::el_mult(parent[0]->output, parent[1]->output, output, 0);
    else {

        Tensor::copy(parent[0]->output, output);
        output->mult_(val);
    }
}

void LMult::backward() {
    // Reserve parent's delta 1
    if (parent[0]->mem_level) { parent[0]->mem_delta(); }

    if (binary) {
        // Reserve parent's delta 2
        if (parent[1]->mem_level) { parent[1]->mem_delta(); }  // TODO: Review!!

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
        n = new LMult(p[0], p[1], "share_" + to_string(c) + name, todev, this->mem_level);
    else
        n = new LMult(p[0], val, "share_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
