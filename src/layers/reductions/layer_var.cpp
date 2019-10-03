
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

    input=l->output;
    output=l->output;
    delta=l->delta;

    this->axis=axis;
    this->keepdims=keepdims;

    // create a sub-graph
    LRMean *m1=new LRMean(this, axis, true,this->name+"mean_keepdims",dev);
    LDiff *diff=new LDiff(this, m1,this->name+"diff",dev);
    LMult *mult=new LMult(diff,diff,this->name+"mult",dev);
    LRMean *m2=new LRMean(mult, axis,keepdims,this->name+"mean_red",dev);
    layers.push_back(m1);
    layers.push_back(diff);
    layers.push_back(mult);
    layers.push_back(m2);

    // detach from the main graph
    detach(m1);
    detach(diff);
    ////////////////////////////

    output=m2->output;
    delta=m2->delta;

    l->addchild(this);
    addparent(l);

}

void LRVar::resize(int b)
{
  int i;

  for(i=0;i<layers.size();i++) layers[i]->resize(b);

  if (target!=nullptr) target->resize(b);
}

void LRVar::forward(){
  for(int i=0;i<layers.size();i++) {
    layers[i]->forward();
  }
}

void LRVar::backward(){
  for(int i=layers.size()-1;i>=0;i--) layers[i]->backward();
}

Layer *LRVar::share(int c, int bs, vector<Layer *> p) {
  LRVar *n;
  n = new LRVar(p[0], axis, keepdims, "share_" + to_string(c) + name, dev);
  n->orig = this;
  return n;
}

Layer *LRVar::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRVar *n;
    n = new LRVar(p[0], axis, keepdims, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
