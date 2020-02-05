/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_normalization.h"
#include "../reductions/layer_reductions.h"
#include "../operators/layer_operators.h"

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
    delta=div->delta;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LNormMax::resize(int batch){

  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  if (target!=nullptr) target->resize(batch);
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
    LNormMax *n = new LNormMax(p[0], epsilon, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LNormMax::clone(int c, int bs, vector<Layer *> p, int todev) {
    LNormMax *n = new LNormMax(p[0], epsilon, "clone_" + to_string(todev) + name, todev, this->mem_level);
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
