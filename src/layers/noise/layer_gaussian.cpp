//
// Created by Salva Carrión on 2019-05-16.
//


#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "../layer.h"

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
