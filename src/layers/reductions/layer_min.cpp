
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

#include "../operators/layer_operators.h"
#include "layer_reductions.h"


using namespace std;

int LRMin::total_layers = 0;

LRMin::LRMin(Layer *l, vector<int> axis, bool keepdims, string name, int dev): ReductionLayer(name, dev) {
    // TODO: Implement
    if(name.empty()) this->name = "reduction_min" + to_string(++total_layers);

    input=l->output;

    output=l->output;
    delta=l->delta;

    this->axis=axis;
    this->keepdims=keepdims;

    if (keepdims){
      os=input->shape;
    }
    else {
      for(int i=0;i<input->ndim;i++) {
        if (find(axis.begin(), axis.end(), i) == axis.end())
            os.push_back(input->shape[i]);
      }
    }

    output=new Tensor(os,dev);
    delta=new Tensor(os,dev);

    l->addchild(this);
    addparent(l);
}

void LRMin::forward(){
    // TODO: Implement
}

void LRMin::backward(){
  // TODO: Implement
}

Layer *LRMin::share(int c, int bs, vector<Layer *> p) {
    // TODO: Implement
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LRMin::clone(int c, int bs, vector<Layer *> p, int todev) {
    // TODO: Implement
    LRMin *n;
    n = new LRMin(p[0], axis, keepdims, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
