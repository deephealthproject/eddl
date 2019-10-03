
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

#ifndef EDDL_LAYER_H
#define EDDL_LAYER_H

#include <string>
#include <cstdio>
#include "../tensor/tensor.h"
#include "../tensor/tensor_reduction.h"
#include "../tensor/nn/tensor_nn.h"

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
    bool isplot;
    bool inner;

    Layer(string name, int dev);



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
