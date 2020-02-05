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

int LNormMinMax::total_layers = 0;


LNormMinMax::LNormMinMax(Layer *parent, float epsilon, string name, int dev, int mem) : LinLayer(name, dev, mem) {

    vector<int> axis;
    if (parent->output->ndim == 2) axis.push_back(1);
    else if (parent->output->ndim == 4) {axis.push_back(1);axis.push_back(2);axis.push_back(3);}
    else msg("LNormMinMax only works over 2D or 4D tensors", "LNormMinMax");

    if(name.empty()) this->name = "normminmax" + to_string(++total_layers);

    this->epsilon = epsilon;

    //
    input=parent->output;

    // create a sub-graph


    // max
    Layer *max=new LRMax(parent, axis, true,this->name+"max",dev);
    Layer *min=new LRMin(parent, axis, true,this->name+"max",dev);
    Layer *maxmin= new LDiff(max,min,this->name+"maxmin",dev);
    Layer *dmin=new LDiff(parent,min,this->name+"dmin",dev);

    Layer *meps=new LSum(maxmin,epsilon,this->name+"sum_eps",dev);
    // norm
    Layer *div=new LDiv(dmin,meps,this->name+"div",dev);

    layers.push_back(max);
    layers.push_back(min);
    layers.push_back(maxmin);
    layers.push_back(meps);
    layers.push_back(dmin);
    layers.push_back(div);

    // detach from the main graph
    parent->detach(max);
    parent->detach(min);
    parent->detach(dmin);

    ////////////////////////////

    output=div->output;
    delta=div->delta;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LNormMinMax::resize(int batch){

  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  if (target!=nullptr) target->resize(batch);
}

void LNormMinMax::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
}

void LNormMinMax::forward() {
  for(int i=0;i<layers.size();i++) {
    layers[i]->forward();
  }
}

void LNormMinMax::backward() {

  for(int i=layers.size()-1;i>=0;i--) {
    layers[i]->backward();
  }

}



Layer *LNormMinMax::share(int c, int bs, vector<Layer *> p) {
    LNormMinMax *n = new LNormMinMax(p[0], epsilon, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LNormMinMax::clone(int c, int bs, vector<Layer *> p, int todev) {
    LNormMinMax *n = new LNormMinMax(p[0], epsilon, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LNormMinMax::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
