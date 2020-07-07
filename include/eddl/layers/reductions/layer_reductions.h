/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#ifndef EDDL_LAYER_REDUCTIONS_H
#define EDDL_LAYER_REDUCTIONS_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

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
    vector<int> axis;
    bool keepdims;

    ReductionLayer(string name, int dev, int mem);

    ~ReductionLayer();

    void mem_delta() override;

    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    string plot(int c) override;


};

class ReductionLayer2 : public Layer {
public:

    int binary;
    float val;
    ReduceDescriptor2 *RD2;
    vector<int> axis;
    bool keepdims;

    ReductionLayer2(string name, int dev, int mem);

    ~ReductionLayer2();


    void addchild(Layer *l) override;

    void addparent(Layer *l) override;

    string plot(int c) override;


};


/// Mean Layer
class LRMean : public ReductionLayer {
public:
    static int total_layers;


    LRMean(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

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

    vector<Layer *> layers;

    LRVar(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

    ~LRVar();

    void forward() override;

    void backward() override;

    void mem_delta() override;
    void free_delta() override;

    void resize(int b) override;

    void reset() override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Sum Layer
class LRSum : public ReductionLayer {
public:
    static int total_layers;

    LRSum(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

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

    LRMax(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

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

    LRMin(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

/// Argmax Layer
class LRArgmax : public ReductionLayer2 {
public:
    static int total_layers;


    LRArgmax(Layer *l, vector<int> axis, bool keepdims, string name, int dev, int mem);

    void forward() override;

    void backward() override;

    void resize(int b) override;

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
};

#endif //EDDL_LAYER_OPERATORS_H
