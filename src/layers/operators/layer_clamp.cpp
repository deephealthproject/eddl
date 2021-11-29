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

int LClamp::total_layers = 0;


LClamp::LClamp(Layer *l, float min, float max, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "clamp_" + to_string(++total_layers);

    this->min=min;
    this->max=max;

    input = l->output;
    output = Tensor::empty_like(input);

    l->addchild(this);
    addparent(l);
}

void LClamp::forward() {
    Tensor::clamp(parent[0]->output, output, this->min, this->max);
}

void LClamp::backward() {
    Tensor::d_clamp(delta, input, parent[0]->delta, this->min, this->max);
}

Layer *LClamp::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LClamp::clone(int c, int bs, vector<Layer *> p, int todev) {
  auto *n = new LClamp(p[0], this->min, this->max, name, todev, this->mem_level);
  n->orig = this;
  return n;
}
