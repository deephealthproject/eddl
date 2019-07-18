
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

#include "layer_merge.h"


using namespace std;

int LMatMul::total_layers = 0;

LMatMul::LMatMul(vector<Layer *> parent, string name, int dev) : MLayer(name, dev) {
    if (parent.size() == 0) msg("Error: LMatMul layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::eqsize(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LMatMul layers with different tensor shape");
            }

    if(name.empty()) this->name = "matmul" + to_string(++total_layers);

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->getShape(), dev);
    delta = new Tensor(parent[0]->output->getShape(), dev);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LMatMul::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LMatMul::forward() {
    // TODO: Implement
    output->set(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LMatMul::backward() {
    // TODO: Implement
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(delta, parent[i]->delta);
}

Layer *LMatMul::share(int c, int bs, vector<Layer *> p) {
    LMatMul *n = new LMatMul(p, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}


Layer *LMatMul::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMatMul *n = new LMatMul(p, "share_" + to_string(c) + name, todev);
    n->orig = this;

    return n;
}





///////////////
