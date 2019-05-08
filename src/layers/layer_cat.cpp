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
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;

int LCat::cat_created = 0;

LCat::LCat(vector<Layer *> parent, string name, int d) : MLayer(name, d) {
    if (parent.size() == 0) msg("Error: LCat layer with empty list");

    dim = parent[0]->output->dim;

    if (parent.size() > 1) {
        for (int i = 0; i < parent.size() - 1; ++i)
            if (dim != parent[i]->output->dim)
                msg("Error: LCat layers with different tensor dims");

        if (dim == 2) {
            for (int i = 0; i < parent.size() - 1; ++i)
                if (parent[i]->output->sizes[0] != parent[i + 1]->output->sizes[0])
                    msg("Error: LCat layers with different size in dim 1");
        } else if (dim == 4) {
            for (int i = 0; i < parent.size() - 1; ++i) {
                if (parent[i]->output->sizes[0] != parent[i + 1]->output->sizes[0])
                    msg("Error: LCat layers with different size in dim 1");
                else if (parent[i]->output->sizes[2] != parent[i + 1]->output->sizes[2])
                    msg("Error: LCat layers with different size in dim 3, rows of 4D");
                else if (parent[i]->output->sizes[3] != parent[i + 1]->output->sizes[3])
                    msg("Error: LCat layers with different size in dim 4, cols of 4D");
            }
        } else {
            msg("Error: LCat layers of 2D or 4D tensors");
        }
    }

    cat_created++;

    input = parent[0]->output;
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->sizes[1];
        index.push_back(t);
    }

    shape s = parent[0]->output->getshape();
    s[1] = t;

    output = new Tensor(s, d);
    delta = new Tensor(s, d);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual
void LCat::forward() {
    int ini = 0;
    for (int i = 0; i < parent.size(); ++i) {
        Tensor::fill(parent[i]->output, 0, parent[i]->output->sizes[1], output, ini, index[i], 0);
        ini = index[i];
    }
}


void LCat::backward() {

    if (parent.size()) {
        int ini = 0;
        for (int i = 0; i < parent.size(); ++i) {
            Tensor::fill(delta, ini, index[i], parent[i]->delta, 0, parent[i]->output->sizes[1], 1);
            ini = index[i];
        }
    }
}


Layer *LCat::share(int c, int bs, vector<Layer *> p) {

    LCat *n = new LCat(p, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LCat::clone(int c, int bs, vector<Layer *> p, int todev) {

    LCat *n = new LCat(p, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LCat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
