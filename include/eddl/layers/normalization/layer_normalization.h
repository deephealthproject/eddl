/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_NORMALIZATION_H
#define EDDL_LAYER_NORMALIZATION_H

#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/regularizers/regularizer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

void BN_forward(Tensor *input, Tensor *bn_mean, Tensor *bn_var, Tensor *mean, Tensor *variance,float momentum, float epsilon, int trmode);
void BN_backward(Tensor *delta, Tensor *bn_var, Tensor *opa);
void rsum(Tensor *A, Tensor *b, Tensor *ones, Tensor *mem,int p=1);
void rdiff(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem,int p=1);
void rmult(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem,int p=1);
void rdiv(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem,int p=1);
void cmean(Tensor *A, Tensor *b,Tensor *ones,int p=1);


class LBatchNorm : public LinLayer {
public:
    float momentum;
    float epsilon;
    bool affine;
    Tensor *mean;
    Tensor *variance;
    Tensor *bn_mean;
    Tensor *bn_var;
    Tensor *bn_g;
    Tensor *bn_b;
    Tensor *gbn_g;
    Tensor *gbn_b;
    Tensor *opa; //output pre-affine

    bool init;
    vector<int> shape;

    static int total_layers;
    vector<Layer *> layers;

    LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void initialize() override;

    void resize(int batch) override;

    int get_trainable_params_count() override;

    string plot(int c) override;
};

/// LayerNormalization Layer
class LLayerNorm : public LinLayer {
public:
    float epsilon;
    bool affine;
    Tensor *mean;
    Tensor *variance;
    Tensor *bn_g;
    Tensor *bn_b;
    Tensor *gbn_g;
    Tensor *gbn_b;
    Tensor *opa; //output pre-affine

    bool init;
    vector<int> shape;

    static int total_layers;
    vector<Layer *> layers;

    LLayerNorm(Layer *parent, float epsilon, bool affine, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void initialize() override;

    void resize(int batch) override;

    int get_trainable_params_count() override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

/// GroupNormalization Layer
class LGroupNorm : public LinLayer {
public:
    float epsilon;
    bool affine;
    int groups;
    Tensor *bn_mean;
    Tensor *bn_var;
    Tensor *bn_g;
    Tensor *bn_b;
    Tensor *gbn_g;
    Tensor *gbn_b;
    Tensor *opa; //output pre-affine

    bool init;
    vector<int> shape;

    static int total_layers;
    vector<Layer *> layers;

    LGroupNorm(Layer *parent, int g,  float epsilon, bool affine,string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void initialize() override;

    void resize(int batch) override;

    int get_trainable_params_count() override;

    string plot(int c) override;
};


/// Normalization Layer
class LNorm : public LinLayer {
public:
    float epsilon;
    static int total_layers;
    vector<Layer *> layers;

    LNorm(Layer *parent, float epsilon,  string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void mem_delta() override;
    void free_delta() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};

class LNormMax : public LinLayer {
public:
    float epsilon;
    static int total_layers;
    vector<Layer *> layers;

    LNormMax(Layer *parent, float epsilon,  string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void mem_delta() override;
    void free_delta() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};


class LNormMinMax : public LinLayer {
public:
    float epsilon;
    static int total_layers;
    vector<Layer *> layers;

    LNormMinMax(Layer *parent, float epsilon,  string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void mem_delta() override;
    void free_delta() override;

    void resize(int batch) override;

    void reset() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_NORMALIZATION_H
