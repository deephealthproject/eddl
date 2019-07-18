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

#include "../operators/layer_operators.h"
#include "layer_reductions.h"


using namespace std;

int LRMean::total_layers = 0;

/**
  @brief Computes the mean of elements across dimensions of a Layer

  @param l a Layer
  @param axis the dimensions to reduce. If NULL (the default), reduces all dimensions
  @param keepdims if true, retains reduced dimensions with length 1. Default False
  @param name a name for the operation (predefined as 'mean+TotaLRMeanLayers')
  @param dev which computing service utilize

  @returns the result of the logarithm operation over l

  Example:
  \verbatim
      # x contains [[1., 1.], [2., 2.]]
      eddl.Mean(x)  # 1.5
      eddl.Mean(x, 0)  # [1.5, 1.5]
      eddl.Mean(x, 1)  # [1.,  2.]
   \endverbatim

  */

LRMean::LRMean(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev):LRMean(l,vector<int>(axis.begin(), axis.end()),keepdims,name,dev){}

LRMean::LRMean(Layer *l, vector <int> axis, bool keepdims, string name, int dev): ReductionLayer(name, dev) {
    if(name.empty()) this->name = "reduction_mean" + to_string(++total_layers);

    input.push_back(l->output);
    this->axis=axis;
    this->keepdims=keepdims;

    if (keepdims){
      os=input[0]->shape;
    }
    else {
      for(int i=0;i<input[0]->ndim;i++) {
        if (find(axis.begin(), axis.end(), i) == axis.end())
            os.push_back(input[0]->shape[i]);
      }
    }

    output=new Tensor(os,dev);
    delta=new Tensor(os,dev);

    l->addchild(this);
    addparent(l);

}

void LRMean::forward(){
    Tensor::reduce(input[0],output,axis,"mean",keepdims,NULL,0);
}

void LRMean::backward(){
    Tensor::delta_reduce(delta,parent[0]->delta,axis,"mean",keepdims,NULL,1);
}

Layer *LRMean::share(int c, int bs, vector<Layer *> p) {
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LRMean::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRMean *n;
    n = new LRMean(p[0], axis, keepdims, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
