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

int LMean::total_layers = 0;

/**
  @brief Computes the mean of elements across dimensions of a Layer

  @param l a Layer
  @param axis the dimensions to reduce. If NULL (the default), reduces all dimensions
  @param keepdims if true, retains reduced dimensions with length 1. Default False
  @param name a name for the operation (predefined as 'mean+TotalMeanLayers')
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
LMean::LMean(Layer *l, int axis, bool keepdims, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "mean" + to_string(++total_layers);
    //TODO: Implement
}


void LMean::forward(){
    //TODO: Implement
}

void LMean::backward(){
    //TODO: Implement
}

Layer *LMean::share(int c, int bs, vector<Layer *> p) {

    return nullptr;
}

Layer *LMean::clone(int c, int bs, vector<Layer *> p, int todev) {

    return nullptr;
}
