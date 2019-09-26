
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

int LActivation::total_layers = 0;

LActivation::LActivation(Layer *parent, string act, string name, int dev) : LinLayer(name, dev) {

    // Set default name
    if(name.empty()) this->name = "activation" + to_string(++total_layers);

    this->act = act;

    input = parent->output;
    output = new Tensor(input->getShape(), dev);
    delta = new Tensor(output->getShape(), dev);
    delta_bp = 0;

    parent->addchild(this);
    addparent(parent);
}
// virtual
void LActivation::resize(int batch){
  Layer::resize(batch);
}


void LActivation::forward() {

    if (act == "relu")
        ReLu(this->input, this->output);
    else if (act == "softmax") {
        Softmax(this->input, this->output);
    }
}


void LActivation::backward() {


    if (parent.size()) {
        if (delta_bp) {
            Tensor::inc(delta, parent[0]->delta);
        } else {
            if (act == "relu")
                D_ReLu(delta, input, parent[0]->delta);
            else if (act == "softmax")
                D_Softmax(delta, output, parent[0]->delta);
        }
    }
}


Layer *LActivation::share(int c, int bs, vector<Layer *> p) {

    LActivation *n = new LActivation(p[0], act, "share_" + to_string(c) + name, dev);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}

Layer *LActivation::clone(int c, int bs, vector<Layer *> p, int todev) {

    LActivation *n = new LActivation(p[0], act, "clone_" + to_string(todev) + name, todev);
    n->orig = this;
    n->delta_bp = delta_bp;

    return n;
}


string LActivation::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightSalmon,shape=box]";

    return s;
}
