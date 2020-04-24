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

int LDiv::total_layers = 0;

/**
  @brief Computes the division operation between two layers

  @param l1 a Layer.
  @param l2 a Layer.
  @param name a name for the operation (predefined as 'div+TotalDivLayers')
  @param dev which computing service utilize

  @returns the result of l1/l2 element-wise

  */
LDiv::LDiv(Layer *l1, Layer *l2, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "div_" + to_string(++total_layers);
    binary = 1;

    input=l1->output;

    output = new Tensor(l1->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(l1->output->shape, dev);  }

    l1->addchild(this);
    l2->addchild(this);
    addparent(l1);
    addparent(l2);
}

/**
  @brief Computes the division operation between a layer and a float

  @param l a Layer.
  @param k a float.
  @param name a name for the operation (predefined as 'div+TotalDivLayers')
  @param dev which computing service utilize

  @returns the result of l/k element-wise over l

  */
LDiv::LDiv(Layer *l, float k, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "div_" + to_string(++total_layers);
    val = k;
    left=1;

    input=l->output;
    output = new Tensor(l->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(l->output->shape, dev);  }

    l->addchild(this);
    addparent(l);
}

LDiv::LDiv(float k, Layer *l, string name, int dev, int mem) : OperatorLayer(name, dev, mem) {
    if(name.empty()) this->name = "div_" + to_string(++total_layers);
    val = k;
    left=0;

    input=l->output;
    output = new Tensor(l->output->shape, dev);
//    if (!mem_level) { delta = new Tensor(l->output->shape, dev);  }

    l->addchild(this);
    addparent(l);
}

void LDiv::forward() {
    if (binary) Tensor::el_div(parent[0]->output, parent[1]->output, output, 0);
    else {
        if (left) {
          Tensor::copy(parent[0]->output, output);
          output->div_(val);
        }
        else {
          Tensor::copy(parent[0]->output, output);
          output->inv_();
          output->mult_(val);
        }
    }
}

void LDiv::backward() {
    if (binary) {
        Tensor::el_div(delta, parent[1]->output, delta, 0);
        Tensor::inc(delta, parent[0]->delta);
        //
        Tensor::el_mult(delta, parent[0]->output, delta, 0);
        Tensor::el_div(delta, parent[1]->output, delta, 0);

        delta->mult_(-1);
        Tensor::inc(delta, parent[1]->delta);
    } else {
      if (left) {
        delta->div_(val);
        Tensor::inc(delta, parent[0]->delta);
      }
      else {
        Tensor::el_div(delta, parent[0]->output, delta, 0);
        Tensor::el_div(delta, parent[0]->output, delta, 0);
        delta->mult_(-val);
        Tensor::inc(delta, parent[0]->delta);
      }
    }
}

Layer *LDiv::share(int c, int bs, vector<Layer *> p) {
  LDiv *n;
  if (binary)
      n = new LDiv(p[0], p[1], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
  else {
    if (left)
      n = new LDiv(p[0], val, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    else
      n = new LDiv(val, p[0], "share_"+to_string(c)+this->name, this->dev, this->mem_level);
  }
  n->orig = this;
  return n;
}

Layer *LDiv::clone(int c, int bs, vector<Layer *> p, int todev) {
    LDiv *n;
    if (binary)
        n = new LDiv(p[0], p[1], "clone_" + to_string(c) + name, todev, this->mem_level);
        else {
          if (left)
            n = new LDiv(p[0], val, "clone_" + to_string(c) + name, todev, this->mem_level);
          else
            n = new LDiv(val, p[0], "clone_" + to_string(c) + name, todev, this->mem_level);
        }
    n->orig = this;
    return n;
}
