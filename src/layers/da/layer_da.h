/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_DA_H
#define EDDL_LAYER_DA_H


#include <string>
#include <stdio.h>

#include "../layer.h"
#include "../../random.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

/// Shift Layer
class LShift : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    string da_mode;
    float constant;

    LShift(Layer *parent, vector<float> factor, string da_mode, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Rotate Layer
class LRotate : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    vector<int> axis;
    bool reshape;
    string da_mode;
    float constant;

    LRotate(Layer *parent, vector<float> factor, vector<int> axis, bool reshape, string da_mode, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Scale Layer
class LScale : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    bool reshape;
    string da_mode;
    float constant;

    LScale(Layer *parent, vector<float> factor, bool reshape, string da_mode, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Flip Layer
class LFlip : public LinLayer {
public:
    static int total_layers;
    int axis;

    LFlip(Layer *parent, int axis, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Crop Layer
class LCrop : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    bool reshape;
    float constant;

    LCrop(Layer *parent, vector<float> factor, bool reshape, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Cutout Layer
class LCutout : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    bool reshape;
    float constant;

    LCutout(Layer *parent, vector<float> factor, bool reshape, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_DA_H
