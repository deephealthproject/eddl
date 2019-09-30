
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

    if (keepdims){
      os=input->shape;
    }
    else {
      for(int i=0;i<input->ndim;i++) {
        if (find(axis.begin(), axis.end(), i) == axis.end())
            os.push_back(input->shape[i]);
      }
    }

    LRMean *m1=new LRMean(l, axis, true,this->name+"mean_keepdims",dev);
    LDiff *diff=new LDiff(l, m1,this->name+"diff",dev);
    LMult *mult=new LMult(diff,diff,this->name+"mult",dev);
    LRMean *m2=new LRMean(mult, axis,keepdims,this->name+"mean_red",dev);


    layers.push_back(m1);
    layers.push_back(diff);
    layers.push_back(mult);
    layers.push_back(m2);

    for(int i=0;i<layers.size();i++) {
      layers[i]->isplot=false;
      layers[i]->inner=true;
    }

    output=m2->output;
    delta=m2->delta;

    l->addchild(this);
    addparent(l);
}

void LRVar::resize(int b)
{
  int i;

  input=parent[0]->output;

  for(i=0;i<layers.size();i++) layers[i]->resize(b);

  output=layers[i-1]->output;
  delta=layers[i-1]->delta;

  if (target!=nullptr) {
     tshape s=target->shape;
     s[0]=b;
     delete target;
     target=new Tensor(s,dev);
   }
}

void LRVar::forward(){
    for(int i=0;i<layers.size();i++) layers[i]->forward();
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
