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


int LEmbedding::total_layers = 0;

LEmbedding::LEmbedding(int input_dim, int output_dim, string name, int dev): LinLayer(name, dev) {
    // TODO: Implement
    if(name.empty()) this->name = "embedding" + to_string(++total_layers);
    this->input_dim = input_dim;
    this->output_dim = output_dim;
}


// virtual ...
void LEmbedding::resize(int batch){
  Layer::resize(batch);
}

string LEmbedding::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}


void LEmbedding::forward() {
    // TODO: Implement
    delta->fill_(0.0);
}


void LEmbedding::backward() {
    // TODO: Implement
}

Layer *LEmbedding::share(int c, int bs, vector<Layer *> p) {
    // TODO: Implement
    LEmbedding *n = new LEmbedding(input_dim, output_dim, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LEmbedding::clone(int c, int bs, vector<Layer *> p, int todev) {
    // TODO: Implement
    LEmbedding *n = new LEmbedding(input_dim, output_dim, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}



//////
