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

#ifndef EDDLL_LAYER_POOL_H
#define EDDLL_LAYER_POOL_H


#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/// Pool2D Layer
class LPool : public LinLayer {
public:
    static int total_layers;
    PoolDescriptor *pd;

    // constructors
    LPool(Layer *parent, PoolDescriptor *cd, string name, int dev);

    void resize(int batch) override;
};

/// MaxPool2D Layer
class LMaxPool : public LPool {
public:

    // constructors and clones
    LMaxPool(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p, string name,
             int d);

    LMaxPool(Layer *parent, const initializer_list<int> &ks, const initializer_list<int> &st,
             const initializer_list<int> &p, string name, int dev);

    LMaxPool(Layer *parent, const vector<int> &ks, const vector<int> &st, string p, string name, int dev);

    LMaxPool(Layer *parent, PoolDescriptor *cd, string name, int dev);

    // Params
    Tensor *indX, *indY;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    string plot(int c) override;

};

/// AveragePool2D Layer
class LAveragePool : public LPool {
public:

    // constructors and clones
    LAveragePool(Layer *parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding, string name, int dev);
    LAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    string plot(int c) override;

};

/// GlobalMaxPool2D Layer
class LGlobalMaxPool : public LPool {
public:

    // constructors and clones
    LGlobalMaxPool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    string plot(int c) override;

};

/// GlobalAveragePool2D Layer
class LGlobalAveragePool : public LPool {
public:

    // constructors and clones
    LGlobalAveragePool(Layer *parent, PoolDescriptor *D, string name, int dev);

//    // Params
//    Tensor *indX, *indY;
//
//    // implementation
//    void forward() override;
//
//    void backward() override;
//
//    Layer *share(int c, int bs, vector<Layer *> p) override;
//
//    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//    string plot(int c) override;

};

#endif //EDDLL_LAYER_POOL_H
