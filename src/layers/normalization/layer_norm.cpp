/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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

int LNorm::total_layers = 0;


LNorm::LNorm(Layer *parent, float epsilon, string name, int dev) : LinLayer(name, dev) {

    vector<int> axis;
    if (parent->output->ndim == 2) axis.push_back(1);
    else if (parent->output->ndim == 4) {axis.push_back(1);axis.push_back(2);axis.push_back(3);}
    else msg("LNorm only works over 2D or 4D tensors", "LNorm");

    if(name.empty()) this->name = "norm" + to_string(++total_layers);

    this->epsilon = epsilon;

    //
    input=parent->output;


    // create a sub-graph
    LRMean *mean_x,*var;
    LMult *mult;
    LSum *veps;
    LDiff *diff;
    LSqrt *sd;
    LDiv *div;

    // mean
    mean_x=new LRMean(parent, axis, true,this->name+"mean_x",dev);

    // var
    diff=new LDiff(parent, mean_x,this->name+"diff",dev);
    mult=new LMult(diff,diff,this->name+"mult",dev);
    var=new LRMean(mult, axis,true,this->name+"mean_mult",dev);
    //sd
    veps=new LSum(var,epsilon,this->name+"sum_eps",dev);
    sd=new LSqrt(veps,this->name+"sqrt",dev);
    // norm
    div=new LDiv(diff,sd,this->name+"div",dev);

    layers.push_back(mean_x); //0
    layers.push_back(diff);  //1 --
    layers.push_back(mult);  //2
    layers.push_back(var);   //3
    layers.push_back(veps);  //4 --
    layers.push_back(sd);    //5 --
    layers.push_back(div);   //6 --

    // detach from the main graph
    parent->detach(mean_x);
    parent->detach(diff);
    ////////////////////////////

    output=div->output;
    delta=div->delta;

    parent->addchild(this);
    addparent(parent);

}


// virtual
void LNorm::resize(int batch){

  for(int i=0;i<layers.size();i++) layers[i]->resize(batch);

  if (target!=nullptr) target->resize(batch);
}

void LNorm::reset()
{
  for (int i = 0; i != layers.size(); i++)
      layers[i]->reset();
}

void LNorm::forward() {
  for(int i=0;i<layers.size();i++) {
    layers[i]->forward();
  }
}

void LNorm::backward() {

  for(int i=layers.size()-1;i>=0;i--) {
    layers[i]->backward();
  }

}



Layer *LNorm::share(int c, int bs, vector<Layer *> p) {
    LNorm *n = new LNorm(p[0], epsilon, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LNorm *n = new LNorm(p[0], epsilon, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
