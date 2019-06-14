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

int LPow::total_layers = 0;

/**
  @brief Computes the power of one layer to another

  @param l1 a Layer.
  @param l2 a Layer.
  @param name a name for the operation (predefined as 'pow+TotalPowLayers')
  @param dev which computing service utilize

  @returns the result of l1^l2

  Example:
  \verbatim
      # x contains [[2, 2], [3, 3]]
      # y contains [[8, 10], [2, 3]]
      eddl.Pow(x, y)  # [[256, 1024], [9, 27]]
   \endverbatim

  */
LPow::LPow(Layer *l1, Layer *l2, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "pow" + to_string(++total_layers);
    //TODO: Implement
}

/**
  @brief Computes the power of one layer to a value

  @param l a Layer.
  @param k a float.
  @param name a name for the operation (predefined as 'diff+TotalDiffLayers')
  @param dev which computing service utilize

  @returns the result of l^k

  Example:
  \verbatim
      # x contains [[1, 2], [3, 4]]
      eddl.Pow(x, 2.0)  # [[1, 4], [9, 16]]
   \endverbatim

  */
LPow::LPow(Layer *l, float k, string name, int dev): OperatorLayer(name, dev) {
    if(name.empty()) this->name = "pow" + to_string(++total_layers);
    //TODO: Implement
}

void LPow::forward(){
    //TODO: Implement
}

void LPow::backward(){
    //TODO: Implement
}

Layer *LPow::share(int c, int bs, vector<Layer *> p) {

    return nullptr;
}

Layer *LPow::clone(int c, int bs, vector<Layer *> p, int todev) {

    return nullptr;
}
