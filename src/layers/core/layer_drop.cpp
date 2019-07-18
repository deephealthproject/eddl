
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

#include "layer_core.h"

using namespace std;

int LDropout::total_layers = 0;

LDropout::LDropout(Layer *parent, float df, string name, int dev) : LinLayer(name, dev) {

    if(name.empty()) this->name = "dropout" + to_string(++total_layers);

    // df: drop factor is the probability to delete (drop) an activation
    this->df = df;

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = new Tensor(input->getShape(), dev);

    mask = new Tensor(input->getShape(), dev);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LDropout::resize(int batch){
  Layer::resize(batch);
  delete mask;
  mask = new Tensor(input->getShape(), dev);
}

void LDropout::forward() {
    if (mode == TRMODE) {
        mask->rand_binary(1.0 - df);
        Tensor::el_mult(input, mask, output, 0);
    } else {
        Tensor::copy(input, output);
        output->mult(1.0 - df);
    }

}

void LDropout::backward() {

    if (parent.size()) {
        Tensor::el_mult(delta, mask, parent[0]->delta, 1);
    }
}


Layer *LDropout::share(int c, int bs, vector<Layer *> p) {

    LDropout *n = new LDropout(p[0], df, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LDropout::clone(int c, int bs, vector<Layer *> p, int todev) {

    LDropout *n = new LDropout(p[0], df, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LDropout::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightpink,shape=box]";

    return s;
}
