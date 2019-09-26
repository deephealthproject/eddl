
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

#include "layer_pool.h"


using namespace std;


// ---- MAXPOOL2D ----
// constructors and clones

// constructors and clones
LMaxPool::LMaxPool(Layer *parent, const vector<int> &ks, const vector<int> &st, string p, string name,
                   int dev) : LMaxPool(parent, new PoolDescriptor(ks, st, p), name, dev) {}

LMaxPool::LMaxPool(Layer *parent, const vector<int> &ks, const vector<int> &st,
               const vector<int> &p, string name, int dev) : LMaxPool(parent, new PoolDescriptor(ks, st, p), name, dev) {}

LMaxPool::LMaxPool(Layer *parent, PoolDescriptor *D, string name, int dev) : LPool(parent, D, name, dev) {
    // Params

    D->indX = new Tensor(D->O->getShape(), dev);
    D->indY = new Tensor(D->O->getShape(), dev);
}


void LMaxPool::resize(int batch){
  //cout<<"Resize "<<name<<"\n";

  LPool::resize(batch);

  delete pd->indX;
  delete pd->indY;

  pd->indX = new Tensor(pd->O->getShape(), dev);
  pd->indY = new Tensor(pd->O->getShape(), dev);

}

void LMaxPool::forward() {
    MPool2D(this->pd);
}

void LMaxPool::backward() {
    // backprop delta
    if (parent.size()) {
        MPool2D_back(this->pd);
    }
}

Layer *LMaxPool::share(int c, int bs, vector<Layer *> p) {
    LMaxPool *n = new LMaxPool(p[0], vector<int>{pd->kr, pd->kc}, vector<int>{pd->sr, pd->sc}, vector<int>{pd->padr, pd->padc},
                           "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LMaxPool::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMaxPool *n = new LMaxPool(p[0], vector<int>{pd->kr, pd->kc}, vector<int>{pd->sr, pd->sc}, vector<int>{pd->padr, pd->padc},
                           "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}

string LMaxPool::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=red,shape=box]";

    return s;
}
