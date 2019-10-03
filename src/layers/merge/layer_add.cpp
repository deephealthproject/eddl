
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


int LAdd::total_layers = 0;



LAdd::LAdd(vector<Layer *> parent, string name, int dev) : MLayer(name, dev) {
    if (parent.size() == 0) msg("Error: LAdd layer with empty list");

    if (parent.size() > 1)
        for (int i = 0; i < parent.size() - 1; ++i)
            if (!Tensor::eqsize(parent[i]->output, parent[i + 1]->output)) {
                parent[i]->output->info();
                parent[i + 1]->output->info();
                msg("Error: LAdd layers with different tensor shape");
            }

    input = parent[0]->output;

    output = new Tensor(parent[0]->output->getShape(), dev);
    delta = new Tensor(parent[0]->output->getShape(), dev);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual

string LAdd::plot(int c) {
    string s;

    s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}


void LAdd::forward() {
    output->set(0.0);
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(parent[i]->output, output);

}

void LAdd::backward() {
    for (int i = 0; i < parent.size(); ++i)
        Tensor::inc(delta, parent[i]->delta);
}

void LAdd::resize(int batch){
  Layer::resize(batch);
}

Layer *LAdd::share(int c, int bs, vector<Layer *> p) {
    LAdd *n = new LAdd(p, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}


Layer *LAdd::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAdd *n = new LAdd(p, "share_" + to_string(c) + name, todev);
    n->orig = this;

    return n;
}





///////////////
