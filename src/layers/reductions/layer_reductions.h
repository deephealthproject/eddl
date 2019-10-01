
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

#ifndef EDDL_LAYER_REDUCTIONS_H
#define EDDL_LAYER_REDUCTIONS_H


#include <string>
#include <stdio.h>

#include "../layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/////////////////////////////////////////
/////////////////////////////////////////
// Reduction layer
class ReductionLayer : public Layer {
public:

    int binary;
    float val;

    ReductionLayer(string name, int dev);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    string plot(int c) override;


};


/// Mean Layer
class LRMean : public ReductionLayer {
public:
    static int total_layers;
    ReduceDescriptor *RD;

    LRMean(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Var Layer
class LRVar : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sum Layer
class LRSum : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRSum(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Max Layer
class LRMax : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRMax(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Min Layer
class LRMin : public ReductionLayer {
public:
    static int total_layers;
    tshape os;
    vector<int> axis;
    bool keepdims;
    Tensor *mean;
    int rsize;
    vector<Layer *> layers;

    LRMin(Layer *l, vector<int> axis, bool keepdims, string name, int dev);

    void forward() override;

    void backward() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

#endif //EDDL_LAYER_OPERATORS_H
