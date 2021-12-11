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

int LSum::total_layers = 0;

/**
  @brief Computes the sum operation between two layers

  @param l1 a Layer.
  @param l2 a Layer.
  @param name a name for the operation (predefined as 'sum+TotalSumLayers')
  @param dev which computing service utilize

  @returns the result of l1+l2 element-wise

  */
LSum::LSum(Layer *l1, Layer *l2, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {


    if(name.empty()) this->name = "sum" + to_string(++total_layers);
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
  @param name a name for the operation (predefined as 'sum+TotalSumLayers')
  @param dev which computing service utilize

  @returns the result of l+k element-wise over l

  */
LSum::LSum(Layer *l, float k, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {

    if(name.empty()) this->name = "sum" + to_string(++total_layers);
    val = k;

    input=l->output;

    output = new Tensor(l->output->shape, dev);

    l->addchild(this);
    addparent(l);
}

void LSum::forward() {
    if (binary) Tensor::add(1.0, parent[0]->output, 1.0, parent[1]->output, output, 0);
    else {
        Tensor::add(parent[0]->output, output, val);
    }
}

void LSum::backward() {
    Tensor::inc(delta, parent[0]->delta);
    if (binary) {
        Tensor::inc(delta, parent[1]->delta);
      }
}

Layer *LSum::share(int c, int bs, vector<Layer *> p) {
  LSum *n;
  if (binary)
      n = new LSum(p[0], p[1],  "share_"+to_string(c)+name, dev, mem_level);
  else
      n = new LSum(p[0], val,  "share_"+to_string(c)+name, dev,mem_level);
  n->orig = this;
  return n;
}

Layer *LSum::clone(int c, int bs, vector<Layer *> p, int todev) {
    LSum *n;
    if (binary)
        n = new LSum(p[0], p[1],  "clone_"+to_string(c)+name, todev,mem_level);
    else
        n = new LSum(p[0], val,  "clone_"+to_string(c)+name, todev,mem_level);
    n->orig = this;
    return n;
}
