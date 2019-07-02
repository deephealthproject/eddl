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

int LAbs::total_layers = 0;

/**
  @brief Computes the absolute value of a Layer

  @param l a Layer.
  @param name a name for the operation (predefined as 'abs+TotalAbsLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */
LAbs::LAbs(Layer *l, string name, int dev): OperatorLayer(name, dev) {
    // Set default name
    if(name.empty()) this->name = "abs" + to_string(++total_layers);

    input.push_back(l->output);

    mask=new Tensor(l->output->getShape(),dev);
    output=new Tensor(l->output->getShape(),dev);
    delta=new Tensor(l->output->getShape(),dev);


    l->addchild(this);
    addparent(l);
}

void LAbs::forward(){
    Tensor::copy(input[0],output);
    output->set_abs();
}

void LAbs::backward(){
    Tensor::sign(input[0],mask);
    Tensor::el_mult(delta,mask,parent[0]->delta,1);
}

Layer *LAbs::share(int c, int bs, vector<Layer *> p) {
    return clone(c,bs,p,dev);
}

Layer *LAbs::clone(int c, int bs, vector<Layer *> p, int todev) {
    LAbs *n = new LAbs(p[0], "share_" + to_string(c) + name, todev);
    n->orig = this;

    return n;
}
