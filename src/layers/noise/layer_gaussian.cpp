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

#include "layer_noise.h"


using namespace std;

int LGaussianNoise::total_layers = 0;

LGaussianNoise::LGaussianNoise(Layer *parent, float stdev, string name, int dev) : LinLayer(name, dev) {
    if (parent->output->ndim != 2) msg("LGaussianNoise only works over 2D tensors", "LGaussianNoise");
    total_layers++;
    this->stdev = stdev;

    // TODO: Implement
}


// virtual
void LGaussianNoise::forward() {
    // TODO: Implement
}

void LGaussianNoise::backward() {
    // TODO: Implement
}


Layer *LGaussianNoise::share(int c, int bs, vector<Layer *> p) {
    LGaussianNoise *n = new LGaussianNoise(p[0], stdev, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LGaussianNoise::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGaussianNoise *n = new LGaussianNoise(p[0], stdev, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LGaussianNoise::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
