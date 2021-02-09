/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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

    ~LPool();

    void mem_delta() override;

    void resize(int batch) override;
};

/// Pool1D Layer
class LPool1D : public LinLayer {
public:
    static int total_layers;
    PoolDescriptor *pd;
    Tensor* input_reshaped;

    // constructors
    LPool1D(Layer *parent, PoolDescriptor *cd, string name, int dev, int mem);

    ~LPool1D();

    void mem_delta() override;

    void resize(int batch) override;
};

/// Pool3D Layer
class LPool3D : public LinLayer {
public:
    static int total_layers;
    PoolDescriptor3D *pd;

    // constructors
    LPool3D(Layer *parent, PoolDescriptor3D *pd, string name, int dev, int mem);

    ~LPool3D();

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

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    string plot(int c) override;

};

/// MaxPool1D Layer
class LMaxPool1D : public LPool1D {
public:

    // constructors and clones
    LMaxPool1D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem);

    LMaxPool1D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, const string& name, int dev, int mem);

    LMaxPool1D(Layer *parent, PoolDescriptor *cd, const string& name, int dev, int mem);

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    string plot(int c) override;

};

/// MaxPool3D Layer
class LMaxPool3D : public LPool3D {
public:

    // constructors and clones
    LMaxPool3D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem);

    LMaxPool3D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, const string& name, int dev, int mem);

    LMaxPool3D(Layer *parent, PoolDescriptor3D *cd, const string& name, int dev, int mem);

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


/// AveragePool1D Layer
class LAveragePool1D : public LPool1D {
public:

    // constructors and clones
    LAveragePool1D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const string& padding, const string& name, int dev, int mem);

    LAveragePool1D(Layer *parent, const vector<int> &pool_size, const vector<int> &strides, const vector<int> &padding, const string& name, int dev, int mem);

    LAveragePool1D(Layer *parent, PoolDescriptor *cd, const string& name, int dev, int mem);

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    string plot(int c) override;

};

#endif //EDDL_LAYER_POOL_H
