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
    vector<int> shift;
    string da_mode;
    float constant;

    LShift(Layer *parent, vector<int> shift, string da_mode, float constant, string name, int dev);

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
    float angle;
    vector<int> axis;
    bool reshape;
    string da_mode;
    float constant;

    LRotate(Layer *parent, float angle, vector<int> axis, bool reshape, string da_mode, float constant, string name, int dev);

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
    vector<int> new_shape;
    bool reshape;
    string da_mode;
    float constant;

    LScale(Layer *parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string name, int dev);

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
    vector<int> from_coords;
    vector<int> to_coords;
    bool reshape;
    float constant;

    LCrop(Layer *parent, vector<int> from_coords, vector<int> to_coords, bool reshape, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


class LCropAndScale : public LCrop {
public:
    static int total_layers;

    LCropAndScale(Layer *parent, vector<int> from_coords, vector<int> to_coords, float constant, string name, int dev);

    void forward() override;

};

/// Cutout Layer
class LCutout : public LinLayer {
public:
    static int total_layers;
    vector<int> from_coords;
    vector<int> to_coords;
    float constant;

    LCutout(Layer *parent, vector<int> from_coords, vector<int> to_coords, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};



/// Shift Layer
class LShiftRandom : public LinLayer {
public:
    static int total_layers;
    vector<float> factor_x;
    vector<float> factor_y;
    string da_mode;
    float constant;

    LShiftRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, string da_mode, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Rotate Layer
class LRotateRandom : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    vector<int> axis;
    string da_mode;
    float constant;

    LRotateRandom(Layer *parent, vector<float> factor, vector<int> axis, string da_mode, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Scale Layer
class LScaleRandom : public LinLayer {
public:
    static int total_layers;
    vector<float> factor;
    string da_mode;
    float constant;

    LScaleRandom(Layer *parent, vector<float> factor, string da_mode, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Flip Layer
class LFlipRandom : public LinLayer {
public:
    static int total_layers;
    int axis;

    LFlipRandom(Layer *parent, int axis, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Crop Layer
class LCropRandom : public LinLayer {
public:
    static int total_layers;
    vector<float> factor_x;
    vector<float> factor_y;
    float constant;

    LCropRandom(Layer *parent,  vector<float> factor_x, vector<float> factor_y, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Crop Layer
class LCropAndScaleRandom : public LCropRandom {
public:
    static int total_layers;

    LCropAndScaleRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, string name, int dev);

    void forward() override;
};


/// Cutout Layer
class LCutoutRandom : public LinLayer {
public:
    static int total_layers;
    vector<float> factor_x;
    vector<float> factor_y;
    float constant;

    LCutoutRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, float constant, string name, int dev);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_DA_H
