/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



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
    ReduceDescriptor *RD;

    ReductionLayer(string name, int dev, int mem=0);

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    string plot(int c) override;


};


/// Mean Layer
class LRMean : public ReductionLayer {
public:
    static int total_layers;


    LRMean(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem=0);

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

    vector<int> axis;
    bool keepdims;

    vector<Layer *> layers;

    LRVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem=0);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    void reset() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sum Layer
class LRSum : public ReductionLayer {
public:
    static int total_layers;

    LRSum(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem=0);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Max Layer
class LRMax : public ReductionLayer {
public:
    static int total_layers;


    LRMax(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem=0);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Min Layer
class LRMin : public ReductionLayer {
public:
    static int total_layers;

    LRMin(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem=0);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

#endif //EDDL_LAYER_OPERATORS_H
