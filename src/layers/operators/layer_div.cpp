
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_operators.h"


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
LDiv::LDiv(Layer *l1, Layer *l2, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "div_" + to_string(++total_layers);
    binary = 1;

    input.push_back(l1->output);
    input.push_back(l2->output);

    output = new Tensor(l1->output->getShape(), dev);
    delta = new Tensor(l1->output->getShape(), dev);

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
LDiv::LDiv(Layer *l, float k, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "div_" + to_string(++total_layers);
    val = k;

    input.push_back(l->output);
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LDiv::forward() {
    if (binary) Tensor::el_div(input[0], input[1], output, 0);
    else {
        Tensor::copy(input[0], output);
        output->div_(val);
    }
}

void LDiv::backward() {
    if (binary) {
        Tensor::el_div(delta, input[1], delta, 0);
        Tensor::inc(delta, parent[0]->delta);
        //
        Tensor::el_div(delta, input[1], delta, 0);
        Tensor::el_mult(delta, input[0], delta, 0);
        delta->mult_(-1);
        Tensor::inc(delta, parent[1]->delta);
    } else {
        delta->div_(val);
        Tensor::inc(delta, parent[0]->delta);
    }
}

Layer *LDiv::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LDiv::clone(int c, int bs, vector<Layer *> p, int todev) {
    LDiv *n;
    if (binary)
        n = new LDiv(p[0], p[1], "share_" + to_string(c) + name, todev);
    else
        n = new LDiv(p[0], val, "share_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
