
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

int LLog::total_layers = 0;

/**
  @brief Computes natural logarithm of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'log_TotalLogLayers')
  @param dev which computing service utilize

  @returns the result of the logarithm operation over l

  */
LLog::LLog(Layer *l, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "log_" + to_string(++total_layers);

    input=l->output;
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LLog::forward() {
    Tensor::copy(parent[0]->output, output);
    output->log_();
}

void LLog::backward() {
  Tensor::el_div(delta,parent[0]->output, parent[0]->delta, 1);
}

Layer *LLog::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LLog::clone(int c, int bs, vector<Layer *> p, int todev) {
  LLog *n;
  n = new LLog(p[0], "share_" + to_string(c) + name, todev);
  n->orig = this;
  return n;
}
