//
// Created by Salva Carrión on 2019-05-16.
//


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"

using namespace std;

int LBatchNorm::total_layers = 0;

LBatchNorm::LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev) : LinLayer(name, dev) {
    if (parent->output->ndim != 2) msg("LBatchNorm only works over 2D tensors", "LBatchNorm");
    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);
    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    // TODO: Implement
}


// virtual
void LBatchNorm::resize(int batch){
  Layer::resize(batch);
}

void LBatchNorm::forward() {
    // TODO: Implement
}

void LBatchNorm::backward() {
    // TODO: Implement
}


Layer *LBatchNorm::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LBatchNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LBatchNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
