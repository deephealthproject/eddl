
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

int LAbs::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotalAbsLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LAbs::LAbs(Layer *l, string name, int dev): OperatorLayer(name, dev) {
    // Set default name
    if(name.empty()) this->name = "abs" + to_string(++total_layers);

    input.push_back(l->output);

    mask=new Tensor(l->output->getShape(),dev);
    output=new Tensor(l->output->getShape(),dev);
    delta=new Tensor(l->output->getShape(),dev);


    l->addchild(this);
    addparent(l);
}

void LAbs::forward(){
    Tensor::copy(input[0],output);
    output->set_abs();
}

void LAbs::backward(){
    Tensor::sign(input[0],mask);
    Tensor::el_mult(delta,mask,parent[0]->delta,1);
}

Layer *LAbs::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LAbs::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAbs *n = new LAbs(p[0], "share_" + to_string(c) + name, todev);
    n->orig = this;

    return n;
}
