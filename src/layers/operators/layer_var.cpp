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

int LVar::total_layers = 0;

LVar::LVar(Layer *l, initializer_list<int> &axis, string name, int dev):LVar(l,vector<int>(axis.begin(), axis.end()),name,dev){}



LVar::LVar(Layer *l, vector<int> axis, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "var" + to_string(++total_layers);

    input.push_back(l->output);

    output=l->output;
    delta=l->delta;

    this->axis=axis;

    rsize=1;
    for(int i=0;i<input[0]->ndim;i++) {
      if (find(axis.begin(), axis.end(), i) == axis.end())
          os.push_back(input[0]->shape[i]);
      else rsize*=input[0]->shape[i];
    }

    LMean *m1=new LMean(this,axis, true,name+"mean_keepdims",dev);
    LDiff *diff=new LDiff(this,m1,name+"diff",dev);
    LMult *mult=new LMult(diff,diff,name+"mult",dev);
    LMean *m2=new LMean(mult,axis,false,name+"mean_red",dev);

    layers.push_back(m1);
    layers.push_back(diff);
    layers.push_back(mult);
    layers.push_back(m2);

    output=m2->output;
    delta=m2->delta;

    l->addchild(this);
    addparent(l);
}

void LVar::forward(){
    for(int i=0;i<layers.size();i++) layers[i]->forward();
}

void LVar::backward(){
  for(int i=layers.size()-1;i>=0;i--) layers[i]->backward();
}

Layer *LVar::share(int c, int bs, vector<Layer *> p) {
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LVar::clone(int c, int bs, vector<Layer *> p, int todev) {
    LVar *n;
    n = new LVar(p[0], axis, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
