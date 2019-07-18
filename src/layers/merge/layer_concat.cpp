// The MIT License (MIT)
//
// Copyright (c) 2019 PRHLT Research Group. Inc. http://www.prhlt.upv.es
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


/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_merge.h"


using namespace std;

int LConcat::total_layers = 0;

LConcat::LConcat(vector<Layer *> parent, string name, int dev) : MLayer(name, dev) {
    if (parent.size() == 0) msg("Error: LConcat layer with empty list");

    ndim = parent[0]->output->ndim;

    if (parent.size() > 1) {
        for (int i = 0; i < parent.size() - 1; ++i)
            if (ndim != parent[i]->output->ndim)
                msg("Error: LConcat layers with different tensor dims");

        if (ndim == 2) {
            for (int i = 0; i < parent.size() - 1; ++i)
                if (parent[i]->output->shape[0] != parent[i + 1]->output->shape[0])
                    msg("Error: LConcat layers with different size in dim 1");
        } else if (ndim == 4) {
            for (int i = 0; i < parent.size() - 1; ++i) {
                if (parent[i]->output->shape[0] != parent[i + 1]->output->shape[0])
                    msg("Error: LConcat layers with different size in dim 1");
                else if (parent[i]->output->shape[2] != parent[i + 1]->output->shape[2])
                    msg("Error: LConcat layers with different size in dim 3, rows of 4D");
                else if (parent[i]->output->shape[3] != parent[i + 1]->output->shape[3])
                    msg("Error: LConcat layers with different size in dim 4, cols of 4D");
            }
        } else {
            msg("Error: LConcat layers of 2D or 4D tensors");
        }
    }

    if(name.empty()) this->name = "concat" + to_string(++total_layers);

    input = parent[0]->output;
    int t = 0;
    for (int i = 0; i < parent.size(); ++i) {
        t += parent[i]->output->shape[1];
        index.push_back(t);
    }

    vector<int> shape = parent[0]->output->getShape();
    shape[1] = t;

    output = new Tensor(shape, dev);
    delta = new Tensor(shape, dev);

    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual
void LConcat::forward() {
    int ini = 0;
    for (int i = 0; i < parent.size(); ++i) {
        Tensor::fill(parent[i]->output, 0, parent[i]->output->shape[1], output, ini, index[i], 0);
        ini = index[i];
    }
}


void LConcat::backward() {

    if (parent.size()) {
        int ini = 0;
        for (int i = 0; i < parent.size(); ++i) {
            Tensor::fill(delta, ini, index[i], parent[i]->delta, 0, parent[i]->output->shape[1], 1);
            ini = index[i];
        }
    }
}


Layer *LConcat::share(int c, int bs, vector<Layer *> p) {

    LConcat *n = new LConcat(p, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LConcat::clone(int c, int bs, vector<Layer *> p, int todev) {

    LConcat *n = new LConcat(p, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LConcat::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightblue3,shape=box]";

    return s;
}
