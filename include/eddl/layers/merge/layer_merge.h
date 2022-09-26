/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_MERGE_H
#define EDDL_LAYER_MERGE_H

#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/// Add Layer
class LAdd : public MLayer {
public:
    static int total_layers;


    LAdd(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Subtract Layer
class LSubtract : public MLayer {
public:
    static int total_layers;


    LSubtract(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// MatMul Layer
class LMatMul : public MLayer {
public:
    static int total_layers;


    LMatMul(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Average Layer
class LAverage : public MLayer {
public:
    static int total_layers;


    LAverage(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Maximum Layer
class LMaximum : public MLayer {
public:
    static int total_layers;


    LMaximum(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Maximum Layer
class LMinimum : public MLayer {
public:
    static int total_layers;


    LMinimum(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;

};

/// Concat Layer
class LConcat : public MLayer {
public:
    unsigned int axis;
    vector<int> index;
    static int total_layers;

    // constructors and clones
    LConcat(vector<Layer *> in, unsigned int axis, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params

    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

#ifdef cFPGA
    int layer_disabled = 0;
    void disable_layer() {layer_disabled = 1; this->name = "concat (disabled)";}
    void allocate_output_fpga_buffer();
#endif

};

/// Concat Layer
class LDConcat : public MLayer {
public:
    unsigned int axis;
    vector<int> index;
    static int total_layers;

    // constructors and clones
    LDConcat(vector<Layer *> in, unsigned int axis, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // Params

    // implementation
    void forward() override;

    void backward() override;

    string plot(int c) override;

};


#endif //EDDL_LAYER_MERGE_H
