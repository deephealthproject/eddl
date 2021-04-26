/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_AUXILIAR_H
#define EDDL_LAYER_AUXILIAR_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

using namespace std;

/// Shape Layer
class LShape : public LinLayer {
public:
    static int total_layers;

    LShape(Layer *parent, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Where Layer
class LWhere : public LinLayer {
public:
    static int total_layers;
    Tensor* condition;

    LWhere(Layer *parent, Layer *condition, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// Equal Layer
class LEqual : public LinLayer {
public:
    static int total_layers;
    Tensor *A;
    Tensor *B;

    LEqual(Layer *parent1, Layer *parent2, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// ConstOfTensor Layer
class LConstOfTensor : public LinLayer {
public:
    static int total_layers;
    Tensor *const_tensor;

    LConstOfTensor(Tensor* const_tensor, string name, int dev, int mem);
    ~LConstOfTensor() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void free_delta() override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// Gather Layer
class LGather : public LinLayer {
public:
    static int total_layers;
    int axis;
    Tensor *indices;
    GatherDescriptor *sd;

    LGather(Layer *parent, int axis, Tensor* indices, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// Expand Layer
class LExpand : public LinLayer {
public:
    static int total_layers;
    int size;
    ExpandDescriptor *sd;

    LExpand(Layer *parent, int size, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_AUXILIAR_H
