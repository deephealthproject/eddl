/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/operators/layer_operators.h"


using namespace std;

int LLog2::total_layers = 0;


/**
  @brief Computes logarithm with base 2 of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'log2_TotalLog2Layers')
  @param dev which computing service utilize

  @returns the result of the logarithm with base 2 operation over l

  */
LLog2::LLog2(Layer *l, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "log2_" + to_string(++total_layers);

    input=l->output;
    output = new Tensor(l->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(l->output->shape, dev);  }

    l->addchild(this);
    addparent(l);
}

void LLog2::forward() {
    Tensor::copy(parent[0]->output, output);
    output->log2_();
}

void LLog2::backward() {
  delta->div_(log(2));
  Tensor::el_div(delta,parent[0]->output, parent[0]->delta, 1);
}

Layer *LLog2::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LLog2::clone(int c, int bs, vector<Layer *> p, int todev) {
  LLog *n;
  n = new LLog(p[0], "share_" + to_string(c) + name, todev, this->mem_level);
  n->orig = this;
  return n;
}
