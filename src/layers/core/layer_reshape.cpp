
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

extern ostream &operator<<(ostream &os, const vector<int> shape);


using namespace std;

int LReshape::total_layers = 0;

LReshape::LReshape(Layer *parent, vector<int> shape, string name, int dev) : LinLayer(name, dev) {
    ls = shape;
    if(name.empty()) this->name = "reshape" + to_string(++total_layers);


    input = parent->output;

    vector<int> sin = input->getShape();
    int tin = input->size;
    int t = 1, c = 0, ind = -1;

    // Check shape comp.
    for (int i = 0; i < ls.size(); i++) {
        if (ls[i] != -1) t *= ls[i];
        else {
            if (c) msg("Ambiguous reshape, more than one -1", "Reshape");
            else {
                c = 1;
                ind = i;
            }
        }
    }

    if (c == 1) {
        if (t > tin) {
            msg("Incompatible shape", "Reshape");
        } else if (tin % t) {
            msg("Incompatible shape", "Reshape");
        } else {
            ls[ind] = tin / t;
            t = tin;
        }
    } else if (t != tin) {
        msg("Incompatible shape", "Reshape");
    }

    ///////

    // sharing the pointers to data
    output = new Tensor(ls, parent->output);
    delta = new Tensor(ls, parent->delta);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LReshape::resize(int batch){
  //cout<<"Resize "<<name<<"\n";

  ls[0]=batch;

  input = parent[0]->output;
  output = new Tensor(ls, parent[0]->output);
  delta = new Tensor(ls, parent[0]->delta);

}

void LReshape::forward() {

}


void LReshape::backward() {

}


Layer *LReshape::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = ls;
    shape[0] = bs;

    LReshape *n = new LReshape(p[0], shape, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LReshape::clone(int c, int bs, vector<Layer *> p, int todev) {
    vector<int> shape = ls;
    shape[0] = bs;

    LReshape *n = new LReshape(p[0], shape, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LReshape::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
