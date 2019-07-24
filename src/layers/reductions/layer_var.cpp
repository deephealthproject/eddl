
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

int LRVar::total_layers = 0;


LRVar::LRVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev): ReductionLayer(name, dev) {
    if(name.empty()) this->name = "reduction_var" + to_string(++total_layers);

    input.push_back(l->output);

    output=l->output;
    delta=l->delta;

    this->axis=axis;
    this->keepdims=keepdims;

    if (keepdims){
      os=input[0]->shape;
    }
    else {
      for(int i=0;i<input[0]->ndim;i++) {
        if (find(axis.begin(), axis.end(), i) == axis.end())
            os.push_back(input[0]->shape[i]);
      }
    }



    LRMean *m1=new LRMean(this, axis, true,name+"mean_keepdims",dev);
    LDiff *diff=new LDiff(this,m1,name+"diff",dev);
    LMult *mult=new LMult(diff,diff,name+"mult",dev);
    LRMean *m2=new LRMean(mult, axis,keepdims,name+"mean_red",dev);

    layers.push_back(m1);
    layers.push_back(diff);
    layers.push_back(mult);
    layers.push_back(m2);

    output=m2->output;
    delta=m2->delta;

    l->addchild(this);
    addparent(l);
}

void LRVar::forward(){
    for(int i=0;i<layers.size();i++) layers[i]->forward();
}

void LRVar::backward(){
  for(int i=layers.size()-1;i>=0;i--) layers[i]->backward();
}

Layer *LRVar::share(int c, int bs, vector<Layer *> p) {
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LRVar::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRVar *n;
    n = new LRVar(p[0], axis, keepdims, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
