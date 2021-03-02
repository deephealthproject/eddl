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

#include "eddl/layers/core/layer_core.h"

using namespace std;


// TODO: Does this even work?

int LTranspose::total_layers = 0;

LTranspose::LTranspose(Layer *parent, vector<int> dims, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "transpose" + to_string(++total_layers);
    this->dims = dims;

    input=parent->output;
    output=new Tensor(input->getShape(),dev);
//    delta=new Tensor(input->getShape(),dev);

    parent->addchild(this);
    addparent(parent);
}

void LTranspose::forward() {
   Tensor::transpose(input, output,dims);
}


void LTranspose::backward() {
   //Tensor::transpose(delta,delta,rdims);
   //Tensor::inc(delta,parent[0]->delta);
}


string LTranspose::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}

Layer *LTranspose::clone(int c, int bs, vector<Layer *> p, int todev) {
  LTranspose *n;
  n = new LTranspose(p[0], dims,  name, todev, this->mem_level);
  n->orig = this;
  return n;
}

Layer *LTranspose::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}
