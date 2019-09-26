
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

int LLog2::total_layers = 0;


/**
  @brief Computes logarithm with base 2 of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'log2_TotalLog2Layers')
  @param dev which computing service utilize

  @returns the result of the logarithm with base 2 operation over l

  */
LLog2::LLog2(Layer *l, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "log2_" + to_string(++total_layers);

    input=l->output;
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LLog2::forward() {
<<<<<<< HEAD
    Tensor::copy(parent[0]->output, output);
    output->log2();
}

void LLog2::backward() {
  delta->div(log(2));
  Tensor::el_div(delta,parent[0]->output, parent[0]->delta, 1);
=======
    Tensor::copy(input[0], output);
    output->log2_();
}

void LLog2::backward() {
    delta->div_(log(2));
  Tensor::el_div(delta,input[0], parent[0]->delta, 1);
>>>>>>> 8f2c1df6d23bf235963a4979296317faf4deee5a
}

Layer *LLog2::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LLog2::clone(int c, int bs, vector<Layer *> p, int todev) {
  LLog *n;
  n = new LLog(p[0], "share_" + to_string(c) + name, todev);
  n->orig = this;
  return n;
}
