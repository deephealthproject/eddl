/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/reductions/layer_reductions.h"


using namespace std;

int LRVar::total_layers = 0;


LRVar::LRVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem) : ReductionLayer(name, dev, mem) {
    if(name.empty()) this->name = "reduction_var" + to_string(++total_layers);

    input=l->output;
    output=l->output;
    this->axis=axis;
    this->keepdims=keepdims;

    // create a sub-graph
    LRMean *m1=new LRMean(this, axis, true,this->name+"mean_keepdims", this->dev, this->mem_level);
    LDiff *diff=new LDiff(this, m1,this->name+"diff", this->dev, this->mem_level);
    LMult *mult=new LMult(diff,diff,this->name+"mult", this->dev, this->mem_level);
    LRMean *m2=new LRMean(mult, axis,keepdims,this->name+"mean_red", this->dev, this->mem_level);
    layers.push_back(m1);
    layers.push_back(diff);
    layers.push_back(mult);
    layers.push_back(m2);

    // detach from the main graph
    detach(m1);
    detach(diff);

    output=m2->output;
//    delta=m2->delta;

    l->addchild(this);
    addparent(l);

}


void LRVar::mem_delta() {
    if(this->delta == nullptr) {

        // Reserve parent's delta AND assign it to this layer
        parent[0]->mem_delta();  // Reserve delta for parent

        // Reserve delta for subops // TODO: Don't like it
        for(auto &l : layers){
            l->mem_delta(); // Reserve delta for m2
        }
        delta=layers[layers.size()-1]->delta; // [m1, diff, mult, m2]

        if(this->verbosity_level >= 2){
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LRVar::free_delta() {
    // Not really needed, but I like to keep all the methods the same (ease the robustness of "copy-paste")
    if(this->delta != nullptr) {

        // Reserve delta for subops // TODO: Don't like it
        for(auto &l : layers){
            l->free_delta(); // Reserve delta for m2
        }
        delta= nullptr;

        if(this->verbosity_level >= 2){
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}


void LRVar::resize(int b)
{
  int i;

  for(i=0;i<layers.size();i++) layers[i]->resize(b);

}

void LRVar::forward(){
  for(int i=0;i<layers.size();i++) {
    layers[i]->forward();
  }
}

void LRVar::backward(){
  for(int i=layers.size()-1;i>=0;i--) { layers[i]->backward(); }
}


void LRVar::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
}

Layer *LRVar::share(int c, int bs, vector<Layer *> p) {
  LRVar *n;
  n = new LRVar(p[0], axis, keepdims, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
  n->orig = this;
  return n;
}

Layer *LRVar::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRVar *n;
    n = new LRVar(p[0], axis, keepdims, "clone_" + to_string(c) + name, todev, this->mem_level);
    n->orig = this;
    return n;
}
