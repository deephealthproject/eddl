/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
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

    ~LDataAugmentation();

    void mem_delta() override;
    void free_delta() override;

//    void resize(int batch) override;

};

/// Shift Layer
class LShift : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> shift;
    string da_mode;
    float constant;

    LShift(Layer *parent, vector<int> shift, string da_mode, float constant, string name, int dev, int mem);

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
    string da_mode;
    float constant;

    LRotate(Layer *parent, float angle, vector<int> offset_center, string da_mode, float constant, string name, int dev, int mem);

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
    string da_mode;
    float constant;

    LScale(Layer *parent, vector<int> new_shape, bool reshape, string da_mode, float constant, string name, int dev, int mem);

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
    float constant;

    LCrop(Layer *parent, vector<int> from_coords, vector<int> to_coords, bool reshape, float constant, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};


class LCropScale : public LCrop {
public:
    static int total_layers;
    string da_mode;

    LCropScale(Layer *parent, vector<int> from_coords, vector<int> to_coords, string da_mode, float constant, string name, int dev, int mem);

    void forward() override;

};

/// Cutout Layer
class LCutout : public LDataAugmentation {
public:
    static int total_layers;
    vector<int> from_coords;
    vector<int> to_coords;
    float constant;

    LCutout(Layer *parent, vector<int> from_coords, vector<int> to_coords, float constant, string name, int dev, int mem);

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
    string da_mode;
    float constant;

    LShiftRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, string da_mode, float constant, string name, int dev, int mem);

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
    string da_mode;
    float constant;

    LRotateRandom(Layer *parent, vector<float> factor, vector<int> offset_center, string da_mode, float constant, string name, int dev, int mem);

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
    string da_mode;
    float constant;

    LScaleRandom(Layer *parent, vector<float> factor, string da_mode, float constant, string name, int dev, int mem);

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
    string da_mode;

    LCropScaleRandom(Layer *parent, vector<float> factor, string da_mode, string name, int dev, int mem);

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
    float constant;

    LCutoutRandom(Layer *parent, vector<float> factor_x, vector<float> factor_y, float constant, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_DA_H
