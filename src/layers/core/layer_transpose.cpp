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
#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"

using namespace std;

int LTranspose::total_layers = 0;

LTranspose::LTranspose(Layer *parent, const initializer_list<int> &dims, string name, int dev):LTranspose(parent, vector<int>(dims.begin(), dims.end()), name,dev){}

LTranspose::LTranspose(Layer *parent, vector<int> dims, string name, int dev) : LinLayer(name, dev) {
    if(name.empty()) this->name = "transpose" + to_string(++total_layers);
    this->dims = dims;

    input=parent->output;
    output=new Tensor(input->getShape(),dev);
    delta=new Tensor(input->getShape(),dev);

    parent->addchild(this);
    addparent(parent);
}


void LTranspose::forward() {
   Tensor::transpose(input,output,dims);
}


void LTranspose::backward() {
   //Tensor::transpose(delta,delta,rdims);
   //Tensor::inc(delta,parent[0]->delta);
}


string LTranspose::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray75,shape=box]";

    return s;
}

Layer *LTranspose::clone(int c, int bs, vector<Layer *> p, int todev) {
  LTranspose *n;
  n = new LTranspose(p[0], dims, "share_" + to_string(c) + name, todev);
  n->orig = this;
  return n;
}

Layer *LTranspose::share(int c, int bs, vector<Layer *> p) {
  return clone(c,bs,p,dev);
}
