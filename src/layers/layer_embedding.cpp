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


int LEmbedding::total_layers = 0;

LEmbedding::LEmbedding(int input_dim, int output_dim, string name, int dev): LinLayer(name, dev) {
    // TODO: Implement
    total_layers++;
    this->input_dim = input_dim;
    this->output_dim = output_dim;
}


// virtual
string LEmbedding::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=LightBlue,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}


void LEmbedding::forward() {
    // TODO: Implement
    delta->set(0.0);
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
