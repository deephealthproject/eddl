/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "layers/operators/layer_operators.h"


using namespace std;

int LExp::total_layers = 0;

/**
  @brief Computes exponential of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'exp+TotalExpLayers')
  @param dev which computing service utilize

  @returns the result of e^l

  */
LExp::LExp(Layer *l, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "exp_" + to_string(++total_layers);

    input=l->output;
    output = new Tensor(l->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(l->output->shape, dev);  }

    l->addchild(this);
    addparent(l);
}

void LExp::forward() {
    Tensor::copy(parent[0]->output, output);
    output->exp_();

}

void LExp::backward() {
  Tensor::el_mult(delta, output, parent[0]->delta, 1);
}

Layer *LExp::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LExp::clone(int c, int bs, vector<Layer *> p, int todev) {
  LExp *n;
  n = new LExp(p[0], "share_" + to_string(c) + name, todev, this->mem_level);
  n->orig = this;
  return n;
}
