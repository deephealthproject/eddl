/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"

using namespace std;


int LInput::total_layers = 0;

LInput::LInput(Tensor *in, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if(name.empty()) this->name = "input" + to_string(++total_layers);
    input = output = in;
}

LInput::~LInput(){
  if (output!=nullptr) {
    delete output;
    output=nullptr;
  }

}


string LInput::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LInput::free_delta(){

  // DO NOT DELETE DELTA
  // There will be problems with network concatenation
  // [Input1]->[Net1]=>[Input2]->[Net2]->[Cost. func]
  // "=>" is a copyTensor(delta2, delta1)
  // If delta2 is deleted after the backward of Input2, there will be nothing to copy

  delta->fill_(0.0);
}

void LInput::forward() {
  if (parent.size()) {
    Tensor::copy(parent[0]->output,output);
  }
  if(quantization_mode>=3)output->quantize_(quantization_clipping_bits, quantization_rounding_bits, 1);
}


void LInput::backward() {

  if (parent.size()) {
    Tensor::copy(delta,parent[0]->delta);
  }

}

Layer *LInput::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LInput *n = new LInput(new Tensor(shape,dev), "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;
    for(int i=0;i<p.size();i++) {
      p[i]->addchild(n);
      n->addparent(p[i]);
    }

    return n;
}

Layer *LInput::clone(int c, int bs, vector<Layer *> p, int todev) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LInput *n = new LInput(new Tensor(shape, todev),  name, todev, this->mem_level);
    n->orig = this;

    for(int i=0;i<p.size();i++) {
      p[i]->addchild(n);
      n->addparent(p[i]);
    }


    return n;
}



//////
