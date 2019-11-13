/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_H
#define EDDL_LAYER_H

#include <string>
#include <cstdio>

#include "../initializers/initializer.h"

#include "../tensor/tensor.h"
#include "../tensor/tensor_reduction.h"
#include "../tensor/nn/tensor_nn.h"
#include "../regularizers/regularizer.h"


#define TRMODE 1
#define TSMODE 0

using namespace std;

class Net;

class Layer {
public:
    string name;
    Tensor *input;
    Tensor *output;
    Tensor *target;
    Tensor *delta;
    Layer *orig;
    Net *net;

    vector<Tensor *> params;
    vector<Tensor *> gradients;

    vector<Layer *> parent;
    vector<Layer *> child;

    Regularizer *reg;
    Initializer *init;

    int mode;
    int dev;
    int lin, lout;
    int delta_bp;

    Layer(string name, int dev);
    // Destructor
    virtual ~Layer();


    void initialize();


    void save(FILE *fe);
    void load(FILE *fe);

    virtual void info();

    void setmode(int m);
    void detach(Layer *l);
    vector<int> getShape();

    Tensor* getWeights();
    Tensor* setWeights(Tensor bias);

    Tensor* getBias();
    Tensor* setBias(Tensor bias);


    //virtual
    virtual void reset();

    virtual void resize(int batch);

    virtual string plot(int c) { return ""; }

    virtual void addchild(Layer *l) {}

    virtual void addparent(Layer *l) {}

    virtual void forward() {}

    virtual void backward() {}

    virtual Layer *share(int c, int bs, vector<Layer *> p) { return nullptr; }

    virtual Layer *clone(int c, int bs, vector<Layer *> p, int todev) { return nullptr; }



};

Layer* operator+(Layer &l1,Layer &l2);
Layer* operator+(Layer &l1,float l2);
Layer* operator+(float f,Layer &l);

Layer* operator*(Layer &l1,Layer &l2);
Layer* operator*(Layer &l1,float l2);
Layer* operator*(float f,Layer &l);

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

#endif //EDDL_LAYER_H
