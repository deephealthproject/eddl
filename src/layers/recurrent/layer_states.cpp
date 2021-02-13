/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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

int LStates::total_layers = 0;

LStates::LStates(Tensor *in, string name, int dev, int mem): MLayer(name, dev, mem) {

    if(name.empty()) this->name = "State" + to_string(++total_layers);
    
    input = output = in;
    // batch x num_states x dim_states

    for(int i=0;i<input->shape[1];i++) {
      states.push_back(new Tensor({input->shape[0],input->shape[2]},dev));
      delta_states.push_back(new Tensor({input->shape[0],input->shape[2]},dev));
    }
   
    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

void LStates::resize(int batch){
  input->resize(batch);
  for(int i=0;i<states.size();i++) {
    states[i]->resize(batch);
    delta_states[i]->resize(batch);
  }
}


// virtual
void LStates::forward() {
  for(int i=0;i<states.size();i++) {
    Tensor *s = input->select({":",to_string(i),":"}); //batch x 1 x dim_states
    Tensor *s2 = s->reshape({input->shape[0],input->shape[2]});
    Tensor::copy(s2,states[i]);
    delete s2; // because reshape() returns a new Tensor object despite the contents is the same of s
    delete s; // because select() returns a new Tensor object
  }   
}

void LStates::backward() {
}

////


Layer *LStates::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LStates *n = new LStates(new Tensor(shape,dev), "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;
    
    for(int i=0;i<p.size();i++) {
      p[i]->addchild(n);
      n->addparent(p[i]);
    }

    return n;
}

Layer *LStates::clone(int c, int bs, vector<Layer *> p, int todev) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LStates *n = new LStates(new Tensor(shape, todev),  name, todev, this->mem_level);
    n->orig = this;

    for(int i=0;i<p.size();i++) {
      p[i]->addchild(n);
      n->addparent(p[i]);
    }


    return n;
}


string LStates::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
