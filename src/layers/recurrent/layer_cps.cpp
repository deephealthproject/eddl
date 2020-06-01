/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <cstdio>
#include <cstdio>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/recurrent/layer_recurrent.h"


using namespace std;

int LCopyStates::total_layers = 0;

LCopyStates::LCopyStates(vector<Layer *> parent, string name, int dev, int mem): MLayer(name, dev, mem) {

    if(name.empty()) this->name = "CopyState" + to_string(++total_layers);

    for(int i=0;i<parent[0]->states.size();i++) {
      states.push_back(new Tensor(parent[0]->states[i]->shape,dev));
      delta_states.push_back(new Tensor(parent[0]->states[i]->shape,dev));
    }
    output=states[0];
    delta=delta_states[0];
    input=parent[0]->output;

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

void LCopyStates::resize(int batch){
  for(int i=0;i<states.size();i++) {
    states[i]->resize(parent[0]->states[i]->shape[0]);
    delta_states[i]->resize(parent[0]->states[i]->shape[0]);
  }
}


// virtual
void LCopyStates::forward() {
  for(int i=0;i<states.size();i++)
    Tensor::copy(parent[0]->states[i],states[i]);
}

void LCopyStates::backward() {
  for(int i=0;i<states.size();i++)
    Tensor::copy(delta_states[i],parent[0]->delta_states[i]);
}

////


Layer *LCopyStates::share(int c, int bs, vector<Layer *> p) {
    LCopyStates *n = new LCopyStates(p, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;

    return n;
}

Layer *LCopyStates::clone(int c, int bs, vector<Layer *> p, int todev) {
    LCopyStates *n = new LCopyStates(p, "share_"+to_string(c)+this->name, todev, this->mem_level);
    n->orig = this;

    return n;
}


string LCopyStates::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
