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

#include "layer_operators.h"


using namespace std;

int LLog2::total_layers = 0;


/**
  @brief Computes logarithm with base 2 of a Layer element-wise

  @param l a Layer.
  @param name a name for the operation (predefined as 'log2_TotalLog2Layers')
  @param dev which computing service utilize

  @returns the result of the logarithm with base 2 operation over l

  */
LLog2::LLog2(Layer *l, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "log2" + to_string(++total_layers);

    input = l->output;
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LLog2::forward() {
    Tensor::copy(parent[0]->output, output);
    output->set_log2();
}

void LLog2::backward() {
  delta->div(log(2));
  Tensor::el_div(delta,parent[0]->output, parent[0]->delta, 1);
}

Layer *LLog2::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LLog2::clone(int c, int bs, vector<Layer *> p, int todev) {
  LLog *n;
  n = new LLog(p[0], "share_" + to_string(c) + name, todev);
  n->orig = this;
  return n;
}
