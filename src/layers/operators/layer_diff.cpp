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

int LDiff::total_layers = 0;

/**
  @brief Computes the subtraction operation between two layers

  @param l1 a Layer.
  @param l2 a Layer.
  @param name a name for the operation (predefined as 'diff+TotalDiffLayers')
  @param dev which computing service utilize

  @returns the result of l1-l2 element-wise

  */
LDiff::LDiff(Layer *l1, Layer *l2, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "diff_" + to_string(++total_layers);
    binary=1;

    input=l1->output;
    tin.push_back(l1->output);
    tin.push_back(l2->output);

    output = new Tensor(l1->output->shape, dev);

    l1->addchild(this);
    l2->addchild(this);
    addparent(l1);
    addparent(l2);
}

/**
  @brief Computes the subtraction operation between a layer and a float

  @param l a Layer.
  @param k a float.
  @param name a name for the operation (predefined as 'diff+TotalDiffLayers')
  @param dev which computing service utilize

  @returns the result of l-k element-wise over l

  */
LDiff::LDiff(Layer *l, float k, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "diff" + to_string(++total_layers);
    val=k;
    left=1;

    input=l->output;

    output = new Tensor(l->output->shape, dev);


    l->addchild(this);
    addparent(l);
}

LDiff::LDiff(float k, Layer *l, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "diff" + to_string(++total_layers);
    val=k;
    left=0;

    input=l->output;


    output = new Tensor(l->output->shape, dev);


    l->addchild(this);
    addparent(l);
}

LDiff::~LDiff(){
}

void LDiff::forward(){

    if (binary) {
      Tensor::add(1.0, tin[0], -1.0, tin[1], output, 0);
    }
    else {
      if (left) {
        Tensor::sub(parent[0]->output, output, val);
      }
      else {
        Tensor::copy(parent[0]->output,output);
        output->mult_(-1);
        output->add_(val);
      }
    }
}

void LDiff::backward(){
    if (binary) {
        Tensor::inc(delta, parent[0]->delta);
        delta->mult_(-1.0);
        Tensor::inc(delta, parent[1]->delta);
    } else {
      if (left) {
          Tensor::inc(delta, parent[0]->delta);
      } else {
          delta->mult_(-1.0);
          Tensor::inc(delta, parent[0]->delta);
        }
    }

}

Layer *LDiff::share(int c, int bs, vector<Layer *> p) {
  LDiff *n;
  if (binary)
      n = new LDiff(p[0], p[1], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
  else {
    if (left)
      n = new LDiff(p[0], val, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    else
      n = new LDiff(val, p[0], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
  }
  n->orig = this;
  return n;
}

Layer *LDiff::clone(int c, int bs, vector<Layer *> p, int todev) {
    LDiff *n;
    if (binary)
        n = new LDiff(p[0], p[1], "clone_" + to_string(c) + name, todev, this->mem_level);
        else {
          if (left)
            n = new LDiff(p[0], val, "clone_" + to_string(c) + name, todev, this->mem_level);
          else
            n = new LDiff(val, p[0], "clone_" + to_string(c) + name, todev, this->mem_level);
        }
    n->orig = this;
    return n;
}
