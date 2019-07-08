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

#ifndef EDDLL_LAYER_REDUCTIONS_H
#define EDDLL_LAYER_REDUCTIONS_H


#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/////////////////////////////////////////
/////////////////////////////////////////
// Reduction layer
class ReductionLayer : public Layer {
public:

    vector<Tensor *>input;

    int binary;
    float val;

    ReductionLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;


};


/// Mean Layer
class LRMean : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;

    LRMean(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev);
    LRMean(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Var Layer
class LRVar : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRVar(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev);
    LRVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sum Layer
class LRSum : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRSum(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev);
    LRSum(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Max Layer
class LRMax : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRMax(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev);
    LRMax(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Min Layer
class LRMin : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRMin(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev);
    LRMin(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

#endif //EDDLL_LAYER_OPERATORS_H
