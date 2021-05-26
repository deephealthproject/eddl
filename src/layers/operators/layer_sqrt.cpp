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

#include "eddl/layers/operators/layer_operators.h"

using namespace std;

int LSqrt::total_layers = 0;

/**
  @brief Computes square root of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'sqrt+TotalSqrtLayers')
  @param dev which computing service utilize

  @returns the result of the square root operation over l

  */

  LSqrt::LSqrt(Layer *l, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
      if(name.empty()) this->name = "sqrt_" + to_string(++total_layers);

      input=l->output;
      output = new Tensor(l->output->shape, dev);

      l->addchild(this);
      addparent(l);
  }

  void LSqrt::forward() {
      Tensor::sqrt(parent[0]->output, output);
  }

  void LSqrt::backward() {
    Tensor::el_div(delta, output, delta, 0);
    delta->div_(2.0);
    Tensor::inc(delta, parent[0]->delta);
  }

  Layer *LSqrt::share(int c, int bs, vector<Layer *> p) {
    LSqrt *n;
    n = new LSqrt(p[0], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    return n;
  }

  Layer *LSqrt::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LSqrt(p[0], "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
  }
