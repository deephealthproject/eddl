/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_HLSINF_H
#define EDDL_LAYER_HLSINF_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"
#include "eddl/layers/merge/layer_merge.h"

using namespace std;

/// HLSinf Layer
class LHLSinf : public MLayer {
public:

    int H, W;                    // input data geometry
    int Ichannels;               // input channels
    int Ochannels;               // output channels
    int KH, KW;                  // Filter size
    int SH, SW;                  // stride size
    int PT, PB, PL, PR;          // padding (top, bottom, left, right)
    int enable_relu;    
    float relu_factor;           // relu factor for leaky relu ( = 0 -> ReLU)
    int enable_clipping;
    int min_clip;
    int max_clip;
    int enable_stm;
    int enable_maxp;
    int enable_avgp;
    int enable_shift;
    int pos_shift;
    int dir_shift;
    int enable_batch_norm;
    int enable_add;
    int enable_upscale;
    int dense_operation;

    Tensor *filter= nullptr;
    Tensor *bias= nullptr;
    Tensor *input_add= nullptr;
    Tensor *batch_norm_values= nullptr;

    static int total_layers;

    LHLSinf(Layer * parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, int enable_upscale, int dense_operation, string name, int dev, int mem) ;

    LHLSinf(vector<Layer * >parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_add, int enable_upscale, int dense_operation, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;
    int get_trainable_params_count();


    void forward() override;

    void backward() override;

    string plot(int c) override;
};

#endif //EDDL_LAYER_HLSINF_H