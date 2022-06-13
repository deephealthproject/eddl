/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: December 2021
* Author: GAP Research Group (UPV), contact: jflich@disca.upv.es
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
    int enable_relu;             // Whether ReLu is enabled
    float relu_factor;           // relu factor for leaky relu ( = 0 -> ReLU)
    int enable_clipping;         // Clipping enabled
    int min_clip;                // Minimum value for clipping
    int max_clip;                // Maximum value for clipping
    int enable_stm;              // Enabled STM (Softplus + Tanh + Tensor multiplication)
    int enable_maxp;             // Enabled max pooling
    int enable_avgp;             // Enabled average pooling
    int enable_shift;            // Enabled shift operation
    int pos_shift;               // Number of bits to shift
    int dir_shift;               // Shift direction, left or right
    int enable_batch_norm;       // Enabled batch normalization
    int enable_bn_relu;          // Enabled relu after BN
    float bn_relu_factor;        // relu factor for leaky relu after bn
    int enable_add;              // Enabled Tensor Add operation
    int enable_add_relu;         // Enabled ReLu activation after add operation
    int upscale_factor;          // Upscale factor
    int dense_operation;         // Enabled dense operation to be performed
    int use_weight_buffer;       // The weight buffer must be used
    int first_row_weight_buffer; // First row to be used in the weight buffer
    int weight_buffer_initialized; // Whether the weight buffer has been initialized or not

    Tensor *filter= nullptr;              // Filter tensor
    Tensor *bias= nullptr;                // Bias tensor
    Tensor *input_add= nullptr;           // Add tensor (tensor to be added to output)
    Tensor *batch_norm_values= nullptr;   // Tensor with batch normalization values (four tensors interleaved)

    static int total_layers;              // Layer identifier (local)

    // Constructor with only one parent layer 
    LHLSinf(Layer * parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_bn_relu, float bn_relu_factor, int enable_add, int enable_add_relu, int upscale_factor, int dense_operation, int use_weight_buffer, int first_row_weight_buffer, string name, int dev, int mem) ;

    // Constructor with multiple parent layers
    LHLSinf(vector<Layer * >parent, int h, int w, int ichannels, int ochannels, int kh, int kw, int sh, int sw, int pt, int pb, int pl, int pr,
              int enable_relu, float relu_factor, int enable_clipping, int min_clip, int max_clip, int enable_shift, int pos_shift, int dir_shift, int enable_stm, int enable_maxp,
              int enable_avgp, int enable_batch_norm, int enable_bn_relu, float bn_relu_factor, int enable_add, int enable_add_relu, int upscale_factor, int dense_operation, int use_weight_buffer, int first_row_weight_buffer, string name, int dev, int mem);

    // Methods
    Layer *share(int c, int bs, vector<Layer *> p) override;
    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;
    void resize(int batch) override;
    int get_trainable_params_count();
    void forward() override;
    void backward() override;
    string plot(int c) override;
};

#endif //EDDL_LAYER_HLSINF_H
