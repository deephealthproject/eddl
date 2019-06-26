// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

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


// virtual
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
