
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


int LInput::total_layers = 0;

LInput::LInput(Tensor *in, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "input" + to_string(++total_layers);
    input = output = in;
    delta = new Tensor(input->getShape(), dev);
}


// virtual
void LInput::resize(int batch){
  Layer::resize(batch);
}

string LInput::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}


void LInput::forward() {
    delta->set(0.0);
}


void LInput::backward() {
}

Layer *LInput::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LInput *n = new LInput(new Tensor(shape), "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LInput::clone(int c, int bs, vector<Layer *> p, int todev) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LInput *n = new LInput(new Tensor(shape, todev), "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}



//////
