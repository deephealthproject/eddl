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
#include "layer_generators.h"


using namespace std;

int LUniform::total_layers = 0;

/**
  @brief Draw samples from a uniform distribution

  @param low Lower boundary of the output interval. All values generated will be greater than or equal to low.
  @param high Upper boundary of the output interval. All values generated will be less than high.
  @param name a name for the operation (predefined as 'abs+TotalAbsLayers')
  @param dev which computing service utilize

  @returns the absolute value of each element in l

  */

LUniform::LUniform(float low, float high, initializer_list<int> &size, string name, int dev):LUniform(low,high,vector<int>(size.begin(), size.end()),name,dev){}

LUniform::LUniform(float low, float high, vector<int> size, string name, int dev): GeneratorLayer(name, dev) {
    // TODO: Implement
    if(name.empty()) this->name = "generator_uniform" + to_string(++total_layers);

    this->low=low;
    this->high=high;

    ////////////

}

void LUniform::forward(){
    // TODO: Implement
}

void LUniform::backward(){
  // TODO: Implement
}

Layer *LUniform::share(int c, int bs, vector<Layer *> p) {
    // TODO: Implement
    clone(c,bs,p,dev);
    return nullptr;
}

Layer *LUniform::clone(int c, int bs, vector<Layer *> p, int todev) {
    // TODO: Implement
    LUniform *n;
    n = new LUniform(low, high, size, "clone_" + to_string(c) + name, todev);
    n->orig = this;
    return n;
}
