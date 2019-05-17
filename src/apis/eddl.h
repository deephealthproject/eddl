// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019
//           Roberto Paredes Palacios, <rparedes@dsic.upv.es>
//           Jon Ander Gómez, <jon@dsic.upv.es>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Sofºtware, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#ifndef _EDDL_
#define _EDDL_

#include <initializer_list>
#include <vector>
#include "../net.h"
#include "../callbacks.h"
#include "../initializer.h"
#include "../losses/loss.h"
#include "../metrics/metric.h"
#include <thread>
#include <pthread.h>

#define tensor LTensor*
#define layer Layer*
#define model Net*
#define optimizer optim*
#define callback Callback*
#define initializer Initializer*
#define loss Loss*
#define metric Metric*
#define compserv CompServ*

typedef vector<LTensor *> vltensor;

class EDDL {
public:
    // ---- TENSOR ----
    static tensor T(const initializer_list<int> &shape);//*
    static tensor T(vector<int> shape);
    static tensor T(string fname);

    // ---- TENSOR OPERATIONS ----
    static void div(tensor t, float v);

    // ---- CORE LAYERS ----
    static layer Activation(layer parent, string activation);
    static layer Activation(layer parent, string activation, string name);

    static layer Conv(layer parent, const initializer_list<int> &ks);//Todo: Remove
    static layer Conv(layer parent, const initializer_list<int> &ks, const initializer_list<int> &st, string p);//Todo: Remove
    static layer Conv(layer parent, const initializer_list<int> &ks, const initializer_list<int> &st);//Todo: Remove
    static layer Conv(layer parent, const initializer_list<int> &ks, string p); //Todo: Remove
    static layer Conv(layer parent, int filters, const initializer_list<int> &kernel_size,
                      const initializer_list<int> &strides, string padding, int groups, const initializer_list<int> &dilation_rate,
                      bool use_bias); //Todo: Implement
    static layer Conv(layer parent, int filters, const initializer_list<int> &kernel_size,
                      const initializer_list<int> &strides, string padding, int groups, const initializer_list<int> &dilation_rate,
                      bool use_bias, string name); //Todo: Implement

    static layer ConvT(layer parent, int filters, const initializer_list<int> &kernel_size,
                       const initializer_list<int> &output_padding, string padding, const initializer_list<int> &dilation_rate,
                       const initializer_list<int> &strides, bool use_bias); //Todo: Implement
    static layer ConvT(layer parent, int filters, const initializer_list<int> &kernel_size,
                       const initializer_list<int> &output_padding, string padding, const initializer_list<int> &dilation_rate,
                       const initializer_list<int> &strides, bool use_bias, string name); //Todo: Implement

    static layer Dense(layer parent, int ndim, bool use_bias=true);//*
    static layer Dense(layer parent, int ndim, bool use_bias, string name); //Todo: Implement

    static layer Embedding(int input_dim, int output_dim); //Todo: Implement
    static layer Embedding(int input_dim, int output_dim, string name); //Todo: Implement

    static layer Input(tensor t); // Why initializing from a tensor?
    static layer Input(const initializer_list<int> &shape);
    static layer Input(const initializer_list<int> &shape, string name);

    // TODO: Interpolate, Resize, upsampling (downsampling?)
    static layer UpSampling(layer parent, const initializer_list<int> &size, string interpolation); //Todo: Implement
    static layer UpSampling(layer parent, const initializer_list<int> &size, string interpolation, string name); //Todo: Implement

    static layer Reshape(layer parent, const initializer_list<int> &shape); //*
    static layer Reshape(layer parent, const initializer_list<int> &shape, string name);

    static layer Transpose(layer parent, const initializer_list<int> &dims); //Todo: Implement
    static layer Transpose(layer parent, const initializer_list<int> &dims, string name); //Todo: Implement

    // ---- LOSSES ----
    static loss LossFunc(string type);

    // ---- METRICS ----
    static metric MetricFunc(string type);

    // ---- MERGE LAYERS ----
    static layer Add(const initializer_list<layer> &layers);
    static layer Add(const initializer_list<layer> &layers, string name);

    static layer Average(const initializer_list<layer> &layers); //Todo: Implement
    static layer Average(const initializer_list<layer> &layers, string name); //Todo: Implement

    static layer Concat(const initializer_list<layer> &layers);
    static layer Concat(const initializer_list<layer> &layers, string name);

    static layer MatMul(const initializer_list<layer> &layers); //Todo: Implement
    static layer MatMul(const initializer_list<layer> &layers, string name); //Todo: Implement

    static layer Maximum(const initializer_list<layer> &layers); //Todo: Implement
    static layer Maximum(const initializer_list<layer> &layers, string name); //Todo: Implement

    static layer Minimum(const initializer_list<layer> &layers); //Todo: Implement
    static layer Minimum(const initializer_list<layer> &layers, string name); //Todo: Implement

    static layer Subtract(const initializer_list<layer> &layers); //Todo: Implement
    static layer Subtract(const initializer_list<layer> &layers, string name); //Todo: Implement


    // ---- NOISE LAYERS ----
    static layer GaussianNoise(layer parent, float stddev); //Todo: Implement
    static layer GaussianNoise(layer parent, float stddev, string name); //Todo: Implement


    // ---- NORMALIZATION LAYERS ----
    static layer BatchNormalization(layer parent, float momentum, float epsilon, bool affine); //Todo: Implement
    static layer BatchNormalization(layer parent, float momentum, float epsilon, bool affine, string name); //Todo: Implement

    static layer Dropout(layer parent, float rate);
    static layer Dropout(layer parent, float rate, string name); //Todo: Implement

    // ---- OPTIMIZERS ----
    static optimizer Adadelta(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement
    static optimizer Adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad); //Todo: Implement
    static optimizer Adagrad(float lr, float epsilon, float weight_decay); //Todo: Implement
    static optimizer Adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay); //Todo: Implement
    static optimizer Nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay); //Todo: Implement
    static optimizer RMSprop(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement

    static optimizer SGD(float lr=0.01f, float momentum=0.0f, float weight_decay=0.0f, bool nesterov=false);

    static void change(optimizer optim, const initializer_list<float> &params);


    // ---- POOLING LAYERS ----
    static layer AveragePool(layer parent, const initializer_list<int> &pool_size);
    static layer AveragePool(layer parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding="none");
    static layer AveragePool(layer parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding, string name);

    static layer GlobalMaxPool(layer parent);
    static layer GlobalMaxPool(layer parent, string name); //Todo: Implement

    static layer GlobalAveragePool(layer parent);
    static layer GlobalAveragePool(layer parent, string name); //Todo: Implement

    static layer MaxPool(layer parent, const initializer_list<int> &pool_size);
    static layer MaxPool(layer parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding="none");
    static layer MaxPool(layer parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding, string name);


    // ---- RECURRENT LAYERS ----
    static layer RNN(layer parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional);
    static layer RNN(layer parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name);

    static layer LSTM(layer parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional);
    static layer LSTM(layer parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name);


//    // ---- LR SCHEDULERS ----
//    static callback CosineAnnealingLR(int T_max, float eta_min, int last_epoch); //Todo: Implement
//    static callback ExponentialLR(float gamma, int last_epoch); //Todo: Implement
//    static callback MultiStepLR(const initializer_list<int> &milestones, float gamma, int last_epoch); //Todo: Implement
//    static callback ReduceLROnPlateau(string metric, string mode, float factor, int patience, float threshold, string threshold_mode, int cooldown, float min_lr, float eps); //Todo: Implement
//    static callback StepLR(int step_size, float gamma, int last_epoch); //Todo: Implement

    // ---- INITIALIZERS ----
    static initializer Constant(float value); //Todo: Implement
    static initializer Identity(float gain); //Todo: Implement
    static initializer GlorotNormal(float seed); //Todo: Implement
    static initializer GlorotUniform(float seed); //Todo: Implement
    static initializer RandomNormal(float mean, float stdev, int seed); //Todo: Implement
    static initializer RandomUniform(float minval, float maxval, int seed); //Todo: Implement
    static initializer Orthogonal(float gain, int seed); //Todo: Implement


    // ---- COMPUTING SERVICES ----
    compserv CS_CPU(int th);
    compserv CS_GPU(const initializer_list<int> &g);
    compserv CS_FGPA(const initializer_list<int> &f);

    // ---- MODEL METHODS ----
    static model Model(vlayer in, vlayer out);
    static void build(model net, optimizer o, const initializer_list<Loss *> &c, const initializer_list<Metric *> &m);
    static void build(model net, optimizer o, const initializer_list<Loss *> &c, const initializer_list<Metric *> &m, CompServ *cs);
    static void summary(model m);
    static void plot(model m, string fname);
    static void fit(model m, const initializer_list<LTensor *> &in, const initializer_list<LTensor *> &out, int batch, int epochs);
    static void fit(model m, const initializer_list<LTensor *> &in, const initializer_list<LTensor *> &out, int batch, int epochs, const initializer_list<Callback *> &cbs); //Todo: Implement
    static void evaluate(model m, const initializer_list<LTensor *> &in, const initializer_list<LTensor *> &out);
    static model load_model(string fname); //Todo: Implement
    static void save_model(model m, string fname); //Todo: Implement
    static void set_trainable(model m); //Todo: Implement
    static model zoo_models(string model_name); //Todo: Implement

    // ---- LAYER METHODS ----
    static void set_trainable(layer l); //Todo: Implement
    static layer get_layer(model m, string layer_name); //Todo: Implement


    // ---- DATASETS ----
    static void download_mnist();
    static void download_cifar10(); //Todo: Implement
    static void download_cifar100(); //Todo: Implement
};

extern EDDL eddl;

#endif
