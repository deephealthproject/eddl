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

#ifndef _OPTIM_
#define _OPTIM_

#include <stdio.h>
#include <string>
#include <initializer_list>
#include <vector>

#include "../layers/layer.h"

using namespace std;

//shorcuts
//#define SGD new sgd

typedef vector<Layer *> vlayer;
typedef vector<Tensor *> vtensor;

class Optimizer {
public:
    string name;
    vlayer layers;

    Optimizer();

    virtual void setlayers(vlayer l) {}

    virtual void applygrads(int batch) {}

    virtual Optimizer *clone() { return nullptr; }

    virtual void change(const initializer_list<float> &p) {}

};

class sgd : public Optimizer {
public:
    float lr;
    float mu;
    float weight_decay;
    bool nesterov;

    vtensor mT;

    explicit sgd(float lr=0.01f, float momentum=0.0f, float weight_decay=0.0f, bool nesterov=false);

    Optimizer *clone() override;

    void setlayers(vlayer l) override;

    void applygrads(int batch) override;

    void change(const initializer_list<float> &p) override;
};

// ---- Adam ----
class adam: public Optimizer {
public:
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float weight_decay;
    bool amsgrad;

    vtensor mT;

    explicit adam(float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float epsilon=1e-8f, float weight_decay=0.0f, bool amsgrad=false);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const initializer_list<float> &p);
};


// ---- AdaDelta ----
class adadelta : public Optimizer {
public:
    float lr;
    float rho;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit adadelta(float lr=0.01f, float rho=0.95f, float epsilon=1e-8f, float weight_decay=0.0f);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const initializer_list<float> &p);
};

// ---- Adagrad ----
class adagrad : public Optimizer {
public:
    float lr;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit adagrad(float lr=0.01f, float epsilon=1e-8f, float weight_decay=0.0f);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const initializer_list<float> &p);
};

// ---- Adamax ----
class adamax : public Optimizer {
public:
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit adamax(float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float epsilon=1e-8f, float weight_decay=0.0f);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const initializer_list<float> &p);
};

// ---- Nadam ----
class nadam : public Optimizer {
public:
    float lr;
    float beta_1;
    float beta_2;
    float epsilon;
    float schedule_decay;

    vtensor mT;

    explicit nadam(float lr=0.01f, float beta_1=0.9f, float beta_2=0.999f, float epsilon=1e-8f, float schedule_decay=0.004f);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const initializer_list<float> &p);
};

// ---- RMSProp ----
class rmsprop : public Optimizer {
public:
    float lr;
    float rho;
    float epsilon;
    float weight_decay;

    vtensor mT;

    explicit rmsprop(float lr=0.01f, float rho=0.9f, float epsilon=1e-8f, float weight_decay=0.0f);

//    Optimizer *clone();
//
//    void setlayers(vlayer l);
//
//    void applygrads(int batch);
//
//    void change(const initializer_list<float> &p);
};
#endif

//////////
