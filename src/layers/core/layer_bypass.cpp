/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"


using namespace std;

int LBypass::total_layers = 0;

LBypass::LBypass(Layer *parent, string bypass_name, string name, int dev, int mem) : LinLayer(name, dev, mem, "bypass") {
    // The bypass layer cannot simply copy the output and delta, but needs to handle mem_delta() and free_delta().
    if(name.empty()) this->name = "bypass" + to_string(++total_layers);

    // Some vars
    this->bypass_name = bypass_name;

    // Set input
    input = parent->output;
    output= parent->output;

    // Set parent
    parent->addchild(this);
    addparent(parent);
}

LBypass::~LBypass(){
    // All of these pointers point to the parent layer so let the parent layer do the work
    this->input = nullptr;
    this->output = nullptr;
    this->delta = nullptr;
}

void LBypass::mem_delta() {
    parent[0]->mem_delta();
    delta = parent[0]->delta;
}


void LBypass::free_delta() {
    // This is de parent's delta. Do not delete it
    delta = nullptr;
}


void LBypass::forward(){

}

void LBypass::backward(){

}

Layer *LBypass::share(int c, int bs, vector<Layer *> p) {
    auto *n = new LBypass(p[0], this->bypass_name, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    return n;
}

Layer *LBypass::clone(int c, int bs, vector<Layer *> p, int todev) {
    auto *n = new LBypass(p[0], this->bypass_name, this->name, todev, this->mem_level);
    n->orig = this;
    return n;
}


string LBypass::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
