
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

#ifndef EDDL_LAYER_CORE_H
#define EDDL_LAYER_CORE_H

#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

/// Tensor Layer
class LTensor : public LinLayer {
public:

    Tensor *data;

    static int total_layers;

    LTensor(string fname);

    LTensor(vector<int> shape, int dev);

    LTensor(const vector<int> shape, float *fptr,int dev);

    explicit LTensor(Layer *l);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void info() override {}

    void forward() override {}

    void backward() override {}

    void resize(int batch) override;

    string plot(int c) override { return ""; }

    LTensor operator+(LTensor L);


};

/// INPUT Layer
class LInput : public LinLayer {
public:
    static int total_layers;

    LInput(Tensor *in, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// EMBEDDING Layer
class LEmbedding : public LinLayer {
public:
    int input_dim;
    int output_dim;
    static int total_layers;

    LEmbedding(int input_dim, int output_dim, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Dense Layer
class LDense : public LinLayer {
public:
    int ndim;
    bool use_bias;  // TODO: Implement
    static int total_layers;

    LDense(Layer *parent, int ndim, bool use_bias, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params
    Tensor *W;
    Tensor *gW;
    Tensor *bias;
    Tensor *gbias;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Activation Layer
class LActivation : public LinLayer {
public:
    string act;
    static int total_layers;

    LActivation(Layer *parent, string act, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Reshape Layer
class LReshape : public LinLayer {
public:
    static int total_layers;
    vector<int> ls;

    // constructors and clones
    LReshape(Layer *parent, vector<int> shape, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;


    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};

/// Transpose Layer
class LTranspose : public LinLayer {
public:
    static int total_layers;
    vector<int> dims;
    vector<int> rdims;

    // constructors and clones
    LTranspose(Layer *parent, vector<int> dims, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
//
//
//    // implementation
    void forward() override;
//
    void backward() override;
    void resize(int batch) override;

    string plot(int c) override;

};

/// Drop-out Layer
class LDropout : public LinLayer {
public:
    static int total_layers;

    // constructors and clones
    LDropout(Layer *parent, float df, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    float df;
    Tensor *mask;

    // implementation
    void forward() override;

    void backward() override;
    void resize(int batch) override;
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


#endif //EDDL_LAYER_CORE_H
