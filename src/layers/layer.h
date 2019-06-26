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

#ifndef EDDLL_LAYER_H
#define EDDLL_LAYER_H

#include <string>
#include <stdio.h>
#include "../tensor/tensor.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


class Layer {
public:
    string name;
    Tensor *input;
    Tensor *output;
    Tensor *target;
    Tensor *delta;
    Layer *orig;

    vector<Tensor *> params;
    vector<Tensor *> gradients;

    vector<Layer *> parent;
    vector<Layer *> child;

    int mode;
    int dev;
    int lin, lout;
    int delta_bp;

    Layer(string name, int dev);



    void initialize();

    void reset();

    virtual void info();

    void setmode(int m);

    Tensor* getWeights();
    Tensor* setWeights(Tensor bias);

    Tensor* getBias();
    Tensor* setBias(Tensor bias);

    //virtual
    virtual void resize(int batch);

    virtual string plot(int c) { return ""; }

    virtual void addchild(Layer *l) {}

    virtual void addparent(Layer *l) {}

    virtual void forward() {}

    virtual void backward() {}

    virtual Layer *share(int c, int bs, vector<Layer *> p) { return nullptr; }

    virtual Layer *clone(int c, int bs, vector<Layer *> p, int todev) { return nullptr; }

};

/////////////////////////////////////////
/////////////////////////////////////////
// Layers with only one input
class LinLayer : public Layer {
public:

    LinLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    //virtual

    string plot(int c) override { return ""; }

    void resize(int batch) override {}

    void forward() override {}

    void backward() override {}

    Layer *share(int c, int bs, vector<Layer *> p) override { return nullptr; }

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override { return nullptr; }

};

/////////////////////////////////////////
/////////////////////////////////////////
// Layers with several inputs (ADD, CAT,...)
class MLayer : public Layer {
public:

    MLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    //virtual

    string plot(int c) override { return ""; }

    void resize(int batch) override {}

    void forward() override {}

    void backward() override {}

    Layer *share(int c, int bs, vector<Layer *> p) override { return nullptr; }

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override { return nullptr; }

};

#endif //EDDLL_LAYER_H
