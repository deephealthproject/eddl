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

#include "../layer.h"

using namespace std;

int LDiv::total_layers = 0;

LDiv::LDiv(Layer *l1, Layer *l2, string name, int dev) : OperatorLayer(name, dev) {
    total_layers++;
    binary = 1;

    input = l1->output;

    output = new Tensor(l1->output->getShape(), dev);
    delta = new Tensor(l1->output->getShape(), dev);

    l1->addchild(this);
    l2->addchild(this);
    addparent(l1);
    addparent(l2);
}

LDiv::LDiv(Layer *l, float k, string name, int dev) : OperatorLayer(name, dev) {
    total_layers++;
    val = k;

    input = l->output;
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LDiv::forward() {
    if (binary) Tensor::el_div(parent[0]->output, parent[1]->output, output, 0);
    else {
        Tensor::copy(parent[0]->output, output);
        output->div(val);
    }
}

void LDiv::backward() {
    if (binary) {
        Tensor::el_div(delta, parent[1]->output, delta, 0);
        Tensor::inc(delta, parent[0]->delta);
        //
        Tensor::el_div(delta, parent[1]->output, delta, 0);
        Tensor::el_mult(delta, parent[0]->output, delta, 0);
        delta->mult(-1);
        Tensor::inc(delta, parent[1]->delta);
    } else {
        delta->div(val);
        Tensor::inc(delta, parent[0]->delta);
    }
}

Layer *LDiv::share(int c, int bs, vector<Layer *> p) {
    LDiv *n;
    if (binary)
        n = new LDiv(p[0], p[1], "share_" + to_string(c) + name, dev);
    else
        n = new LDiv(p[0], val, "share_" + to_string(c) + name, dev);
    n->orig = this;
    return n;
}

Layer *LDiv::clone(int c, int bs, vector<Layer *> p, int todev) {
    LDiv *n;
    if (binary)
        n = new LDiv(p[0], p[1], "share_" + to_string(c) + name, todev);
    else
        n = new LDiv(p[0], val, "share_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
