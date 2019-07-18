
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

int LExp::total_layers = 0;

/**
  @brief Computes exponential of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'exp+TotalExpLayers')
  @param dev which computing service utilize

  @returns the result of e^l

  */
LExp::LExp(Layer *l, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "exp" + to_string(++total_layers);

    input.push_back(l->output);
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LExp::forward() {
    Tensor::copy(input[0], output);
    output->set_exp();
}

void LExp::backward() {
  Tensor::el_mult(delta, output, parent[0]->delta, 1);
}

Layer *LExp::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LExp::clone(int c, int bs, vector<Layer *> p, int todev) {
  LExp *n;
  n = new LExp(p[0], "share_" + to_string(c) + name, todev);
  n->orig = this;
  return n;
}
