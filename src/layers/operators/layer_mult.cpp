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

#include "layer_operators.h"


using namespace std;

int LMult::total_layers = 0;

LMult::LMult(Layer *l1, Layer *l2, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "mult" + to_string(++total_layers);
    binary = 1;

    input.push_back(l1->output);
    input.push_back(l2->output);

    output = new Tensor(l1->output->getShape(), dev);
    delta = new Tensor(l1->output->getShape(), dev);

    l1->addchild(this);
    l2->addchild(this);
    addparent(l1);
    addparent(l2);
}

/**
  @brief Computes the sum operation between a layer and a float

  @param l a Layer.
  @param k a float.
  @param name a name for the operation (predefined as 'sum+TotaLMultLayers')
  @param dev which computing service utilize

  @returns the result of l+k element-wise over l

  */
LMult::LMult(Layer *l, float k, string name, int dev) : OperatorLayer(name, dev) {
    if(name.empty()) this->name = "mult" + to_string(++total_layers);
    val = k;

    input.push_back(l->output);
    output = new Tensor(l->output->getShape(), dev);
    delta = new Tensor(l->output->getShape(), dev);

    l->addchild(this);
    addparent(l);
}

void LMult::forward() {
    if (binary) Tensor::el_mult(input[0], input[1], output, 0);
    else {
        Tensor::copy(input[0], output);
        output->mult(val);
    }
}

void LMult::backward() {
    if (binary) {
        Tensor::el_mult(delta,input[0],parent[1]->delta,1);
        Tensor::el_mult(delta,input[1],parent[0]->delta,1);
    }
    else {
      delta->mult(val);
      Tensor::inc(delta,parent[0]->delta);
    }
}

Layer *LMult::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}

Layer *LMult::clone(int c, int bs, vector<Layer *> p, int todev) {
    LMult *n;
    if (binary)
        n = new LMult(p[0], p[1], "share_" + to_string(c) + name, todev);
    else
        n = new LMult(p[0], val, "share_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
