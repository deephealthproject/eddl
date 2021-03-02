/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/reductions/layer_reductions.h"
#include "eddl/layers/operators/layer_operators.h"

using namespace std;

int LNormMax::total_layers = 0;


LNormMax::LNormMax(Layer *parent, float epsilon, string name, int dev, int mem) : LinLayer(name, dev, mem) {

    vector<int> axis;
    if (parent->output->ndim == 2) axis.push_back(1);
    else if (parent->output->ndim == 4) {axis.push_back(1);axis.push_back(2);axis.push_back(3);}
    else msg("LNormMax only works over 2D or 4D tensors", "LNormMax");

    if(name.empty()) this->name = "normmax" + to_string(++total_layers);

    this->epsilon = epsilon;

    //
    input=parent->output;

    // create a sub-graph
    LRMax *max;
    LSum *meps;
    LDiv *div;

    // max
    max=new LRMax(parent, axis, true,this->name+"max", this->dev, this->mem_level);

    meps=new LSum(max,epsilon,this->name+"sum_eps", this->dev, this->mem_level);
    // norm
    div=new LDiv(parent,meps,this->name+"div", this->dev, this->mem_level);

    layers.push_back(max);
    layers.push_back(meps);
    layers.push_back(div);

    // detach from the main graph
    parent->detach(max);
    parent->detach(div);

    ////////////////////////////

    output=div->output;
//    delta=div->delta;

    parent->addchild(this);
    addparent(parent);

}

LNormMax::~LNormMax(){
}

void LNormMax::mem_delta() {
    // TEMPORAL!
    if(this->delta == nullptr) {

        // Reserve parent's delta AND assign it to this layer
        parent[0]->mem_delta();  // Reserve delta for parent

        // Reserve delta for subops // TODO: Don't like it
        for(auto &l : layers){
            l->mem_delta(); // Reserve delta for m2
        }
        delta=layers[layers.size()-1]->delta; // Last operation

        if(this->verbosity_level >= 2){
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LNormMax::free_delta() {
    // TEMPORAL!
    // Not really needed, but I like to keep all the methods the same (ease the robustness of "copy-paste")
    if(this->delta != nullptr) {

        // Reserve delta for subops // TODO: Don't like it
        for(auto &l : layers){
            l->free_delta(); // Reserve delta for m2
        }
        delta = nullptr;

        if(this->verbosity_level >= 2){
            std::cout << "Deleted delta for: " + this->name << std::endl;
        }
    }
}


// virtual
void LNormMax::resize(int batch){

  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  
}

void LNormMax::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
}

void LNormMax::forward() {
  for(int i=0;i<layers.size();i++) {
    layers[i]->forward();
  }
}

void LNormMax::backward() {
  for(int i=layers.size()-1;i>=0;i--) {
    layers[i]->backward();
  }

}



Layer *LNormMax::share(int c, int bs, vector<Layer *> p) {
    LNormMax *n = new LNormMax(p[0], epsilon, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LNormMax::clone(int c, int bs, vector<Layer *> p, int todev) {
    LNormMax *n = new LNormMax(p[0], epsilon,  name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LNormMax::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
