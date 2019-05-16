//
// Created by Salva Carrión on 2019-05-16.
//


#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;

int LRNN::total_layers = 0;

LRNN::LRNN(Layer *parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev) : LinLayer(name, dev) {

    this->units = units;
    this->num_layers = num_layers;
    this->use_bias = use_bias;
    this->dropout = dropout;
    this->bidirectional = bidirectional;

    // TODO: Implement
}


// virtual
void LRNN::forward() {
    // TODO: Implement
}

void LRNN::backward() {
    // TODO: Implement
}


Layer *LRNN::share(int c, int bs, vector<Layer *> p) {
    LRNN *n = new LRNN(p[0], units, num_layers, use_bias, dropout, bidirectional, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LRNN::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRNN *n = new LRNN(p[0], units, num_layers, use_bias, dropout, bidirectional, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRNN::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
