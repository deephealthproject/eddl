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

#ifndef EDDLL_LAYER_OPERATORS_H
#define EDDLL_LAYER_OPERATORS_H


#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/////////////////////////////////////////
/////////////////////////////////////////
// Operator layer
class OperatorLayer : public Layer {
public:

    int binary;
    float val;

    OperatorLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;


};

/// Abs Layer
class LAbs : public OperatorLayer {
public:
    static int total_layers;

    Tensor *mask;

    LAbs(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Diff Layer
class LDiff : public OperatorLayer {
public:
    static int total_layers;

    LDiff(Layer *l1, Layer *l2, string name, int dev);
    LDiff(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Div Layer
class LDiv : public OperatorLayer {
public:
    static int total_layers;

    LDiv(Layer *l1, Layer *l2, string name, int dev);
    LDiv(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Exp Layer
class LExp : public OperatorLayer {
public:
    static int total_layers;

    LExp(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Log Layer
class LLog : public OperatorLayer {
public:
    static int total_layers;

    LLog(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Log2 Layer
class LLog2 : public OperatorLayer {
public:
    static int total_layers;

    LLog2(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Log10 Layer
class LLog10 : public OperatorLayer {
public:
    static int total_layers;

    LLog10(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Mean Layer
class LMean : public OperatorLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;

    LMean(Layer *l, initializer_list<int> &axis, bool keepdims, string name, int dev);
    LMean(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Mult Layer
class LMult : public OperatorLayer {
public:
    static int total_layers;

    LMult(Layer *l1, Layer *l2, string name, int dev);
    LMult(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Pow Layer
class LPow : public OperatorLayer {
public:
    static int total_layers;

    LPow(Layer *l1, Layer *l2, string name, int dev);
    LPow(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sqrt Layer
class LSqrt : public OperatorLayer {
public:
    static int total_layers;

    LSqrt(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sum Layer
class LSum : public OperatorLayer {
public:
    static int total_layers;

    LSum(Layer *l1, Layer *l2, string name, int dev);
    LSum(Layer *l, float k, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Var Layer
class LVar : public OperatorLayer {
public:
    static int total_layers;

    LVar(Layer *l, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

#endif //EDDLL_LAYER_OPERATORS_H
