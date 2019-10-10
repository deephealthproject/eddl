/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


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


void LRVar::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
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
