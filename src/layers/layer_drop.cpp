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

int LDrop::drop_created = 0;

LDrop::LDrop(Layer *parent, float df, string name, int d) : LinLayer(name, d) {

    drop_created++;

    // df: drop factor is the probability to delete (drop) an activation
    this->df = df;

    input = parent->output;
    output = new Tensor(input->getshape(), d);
    delta = new Tensor(input->getshape(), d);

    mask = new Tensor(input->getshape(), d);

    parent->addchild(this);
    addparent(parent);
}


// virtual
void LDrop::forward() {
    if (mode == TRMODE) {
        mask->rand_binary(1.0 - df);
        Tensor::el_mult(input, mask, output, 0);
    } else {
        Tensor::copy(input, output);
        output->mult(1.0 - df);
    }

}

void LDrop::backward() {

    if (parent.size()) {
        Tensor::el_mult(delta, mask, parent[0]->delta, 1);
    }
}


Layer *LDrop::share(int c, int bs, vector<Layer *> p) {

    LDrop *n = new LDrop(p[0], df, "share_" + to_string(c) + name, dev);
    n->orig = this;

    return n;
}

Layer *LDrop::clone(int c, int bs, vector<Layer *> p, int todev) {

    LDrop *n = new LDrop(p[0], df, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    return n;
}


string LDrop::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,tshape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=lightpink,tshape=box]";

    return s;
}
