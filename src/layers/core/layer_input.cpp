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

#include "layer_core.h"

using namespace std;


int LInput::total_layers = 0;

LInput::LInput(Tensor *in, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "input" + to_string(++total_layers);
    input = output = in;
    delta = new Tensor(input->getShape(), dev);
}

LInput::~LInput()
{
  if (output!=nullptr) {
    delete output;
    output=nullptr;
  }

}

// virtual
void LInput::resize(int batch){
  Layer::resize(batch);
}

string LInput::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}


void LInput::forward() {

}


void LInput::backward() {

}

Layer *LInput::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LInput *n = new LInput(new Tensor(shape), "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LInput::clone(int c, int bs, vector<Layer *> p, int todev) {
    vector<int> shape = input->getShape();
    shape[0] = bs;

    LInput *n = new LInput(new Tensor(shape, todev), "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}



//////
