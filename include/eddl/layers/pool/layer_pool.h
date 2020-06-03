/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_POOL_H
#define EDDL_LAYER_POOL_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/// Pool2D Layer
class LPool : public LinLayer {
public:
    static int total_layers;
    PoolDescriptor *pd;

    // constructors
    LPool(Layer *parent, PoolDescriptor *cd, string name, int dev, int mem);

    void mem_delta() override;

    void resize(int batch) override;
};

/// MaxPool2D Layer
class LMaxPool : public LPool {
public:

    // constructors and clones
    LMaxPool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem);

    LMaxPool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, const string& name, int dev, int mem);

    LMaxPool(Layer *parent, PoolDescriptor *cd, const string& name, int dev, int mem);

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
    // Params
    Tensor *indX, *indY;

    // constructors and clones
    LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem);

    LAveragePool(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, string name, int dev, int mem);

    LAveragePool(Layer *parent, PoolDescriptor *D, const string& name, int dev, int mem);

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    string plot(int c) override;

};

#endif //EDDL_LAYER_POOL_H
