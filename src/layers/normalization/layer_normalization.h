/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_NORMALIZATION_H
#define EDDL_LAYER_NORMALIZATION_H

#include <string>
#include <stdio.h>

#include "../layer.h"
#include "../core/layer_core.h"
#include "../../regularizers/regularizer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/// Normalization Layer
class LNorm : public LinLayer {
public:
    float epsilon;
    static int total_layers;
    vector<Layer *> layers;

    LNorm(Layer *parent, float epsilon,  string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};

class LNormMax : public LinLayer {
public:
    float epsilon;
    static int total_layers;
    vector<Layer *> layers;

    LNormMax(Layer *parent, float epsilon,  string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};


class LNormMinMax : public LinLayer {
public:
    float epsilon;
    static int total_layers;
    vector<Layer *> layers;

    LNormMinMax(Layer *parent, float epsilon,  string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};

/// BatchNormalization Layer
class LBatchNorm : public LinLayer {
public:
    float momentum;
    float epsilon;
    bool affine;
    LTensor *mean;
    LTensor *variance;
    bool init;

    static int total_layers;
    vector<Layer *> layers;

    LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};

/// BatchNormalization Layer
class LBatchNorm2D : public LinLayer {
public:
    float momentum;
    float epsilon;
    bool affine;
    Tensor *mean;
    Tensor *variance;
    Tensor *bn_mean;
    Tensor *bn_var;
    Tensor *sd;
    Tensor *bn_E;

    int *redmap;
    bool init;
    vector<int> axis;
    vector<int> shape;

    static int total_layers;
    vector<Layer *> layers;

    LBatchNorm2D(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;
};



#endif //EDDL_LAYER_NORMALIZATION_H
