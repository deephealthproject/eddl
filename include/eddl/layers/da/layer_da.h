/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_DA_H
#define EDDL_LAYER_DA_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"
#include "eddl/random.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;

class LDataAugmentation : public LinLayer {
public:
    LDataAugmentation(Layer *parent, string name, int dev, int mem);

    ~LDataAugmentation() override;

    void mem_delta() override;
    void free_delta() override;

//    void resize(int batch) override;

};

/// Shift Layer
class LShift : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> shift;
    WrappingMode da_mode;
    float cval;

    LShift(Layer *parent, vector<int> shift, WrappingMode da_mode, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Rotate Layer
class LRotate : public LDataAugmentation {
public:
    static int total_layers;
    float angle;
    vector<int> offset_center;
    WrappingMode da_mode;
    float cval;

    LRotate(Layer *parent, float angle, vector<int> offset_center, WrappingMode da_mode, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Scale Layer
class LScale : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> new_shape;
    bool reshape;
    WrappingMode da_mode;
    float cval;

    LScale(Layer *parent, vector<int> new_shape, bool reshape, WrappingMode da_mode, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Flip Layer
class LFlip : public LDataAugmentation {
public:
    static int total_layers;
    int axis;

    LFlip(Layer *parent, int axis, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Crop Layer
class LCrop : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> from_coords;
    vector<int> to_coords;
    bool reshape;
    float cval;

    LCrop(Layer *parent, vector<int> from_coords, vector<int> to_coords, bool reshape, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


class LCropScale : public LCrop {
public:
    static int total_layers;
    WrappingMode da_mode;

    LCropScale(Layer *parent, vector<int> from_coords, vector<int> to_coords, WrappingMode da_mode, float cval, string name, int dev, int mem);

    void forward() override;
};

/// Cutout Layer
class LCutout : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> from_coords;
    vector<int> to_coords;
    float cval;

    LCutout(Layer *parent, vector<int> from_coords, vector<int> to_coords, float constant, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Pad Layer
class LPad : public LinLayer {  // Cannot inherit from LDataAugmentation because the backward
public:
    static int total_layers;
    vector<int> padding;
    float constant = 0.0f;

    LPad(Layer *parent, vector<int> padding, float constant, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Shift Layer
class LShiftRandom : public LDataAugmentation {
public:
    static int total_layers;
    vector<float> factor_x;
    vector<float> factor_y;
    WrappingMode da_mode;
    float cval;

    LShiftRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, WrappingMode da_mode, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Rotate Layer
class LRotateRandom : public LDataAugmentation {
public:
    static int total_layers;
    vector<float> factor;
    vector<int> offset_center;
    WrappingMode da_mode;
    float cval;

    LRotateRandom(Layer *parent, vector<float> factor, vector<int> offset_center, WrappingMode da_mode, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Scale Layer
class LScaleRandom : public LDataAugmentation {
public:
    static int total_layers;
    vector<float> factor;
    WrappingMode da_mode;
    float cval;

    LScaleRandom(Layer *parent, vector<float> factor, WrappingMode da_mode, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Flip Layer
class LFlipRandom : public LDataAugmentation {
public:
    static int total_layers;
    int axis;

    LFlipRandom(Layer *parent, int axis, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Crop Layer
class LCropRandom : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> new_shape;

    LCropRandom(Layer *parent,  vector<int> new_shape, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Crop Layer
class LCropScaleRandom : public LDataAugmentation {
public:
    static int total_layers;
    vector<float> factor;
    WrappingMode da_mode;

    LCropScaleRandom(Layer *parent, vector<float> factor, WrappingMode da_mode, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


/// Cutout Layer
class LCutoutRandom : public LDataAugmentation {
public:
    static int total_layers;
    vector<float> factor_x;
    vector<float> factor_y;
    float cval;

    LCutoutRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, float cval, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_DA_H
