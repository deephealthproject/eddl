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
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"

extern ostream &operator<<(ostream &os, const vector<int> shape);


using namespace std;

int LReshape::total_layers = 0;

LReshape::LReshape(Layer *parent, const initializer_list<int> &init, string name, int d) : LReshape(parent, vector<int>(init.begin(), init.end()), "reshape" + to_string(total_layers), d) {}

LReshape::LReshape(Layer *parent, vector<int> shape, string name, int d) : LinLayer(name, d) {
    ls = shape;
    if(name.empty()) this->name = "reshape" + to_string(++total_layers);

    input = parent->output;

    vector<int> sin = input->getShape();
    int tin = input->size;
    int t = 1, c = 0, ind = -1;

    // Check shape comp.
    for (int i = 0; i < ls.size(); i++) {
        if (ls[i] != -1) t *= ls[i];
        else {
            if (c) msg("Ambiguous reshape, more than one -1", "Reshape");
            else {
                c = 1;
                ind = i;
            }
        }
    }

    if (c == 1) {
        if (t > tin) {
            msg("Incompatible shape", "Reshape");
        } else if (tin % t) {
            msg("Incompatible shape", "Reshape");
        } else {
            ls[ind] = tin / t;
            t = tin;
        }
    } else if (t != tin) {
        msg("Incompatible shape", "Reshape");
    }

    ///////

    // sharing the pointers to data
    output = new Tensor(ls, parent->output);
    delta = new Tensor(ls, parent->delta);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LReshape::forward() {

}


void LReshape::backward() {

}


Layer *LReshape::share(int c, int bs, vector<Layer *> p) {
    vector<int> shape = ls;
    shape[0] = bs;

    LReshape *n = new LReshape(p[0], shape, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LReshape::clone(int c, int bs, vector<Layer *> p, int todev) {
    vector<int> shape = ls;
    shape[0] = bs;

    LReshape *n = new LReshape(p[0], shape, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LReshape::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}
