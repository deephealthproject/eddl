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
#include <thread>
#include <pthread.h>

#include "../net.h"
#include "../callbacks/callbacks.h"
#include "../initializers/initializer.h"
#include "../losses/loss.h"
#include "../metrics/metric.h"

#include "../layers/layer.h"
#include "../layers/conv/layer_conv.h"
#include "../layers/core/layer_core.h"
#include "../layers/merge/layer_merge.h"
#include "../layers/noise/layer_noise.h"
#include "../layers/operators/layer_operators.h"
#include "../layers/pool/layer_pool.h"
#include "../layers/recurrent/layer_recurrent.h"

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
    static layer Activation(layer parent, string activation, string name="");

    static layer Conv(layer parent, int filters, const initializer_list<int> &kernel_size,
                      const initializer_list<int> &strides={1,1}, string padding="same", int groups=1, const initializer_list<int> &dilation_rate={1,1},
                      bool use_bias=true, string name=""); //Todo: Implement

    static layer ConvT(layer parent, int filters, const initializer_list<int> &kernel_size,
                       const initializer_list<int> &output_padding, string padding="same", const initializer_list<int> &dilation_rate={1,1},
                       const initializer_list<int> &strides={1,1}, bool use_bias=true, string name=""); //Todo: Implement

    static layer Dense(layer parent, int ndim, bool use_bias=true, string name=""); //Todo: Implement

    static layer Embedding(int input_dim, int output_dim, string name=""); //Todo: Implement

    static layer Input(const initializer_list<int> &shape, string name="");

    // TODO: Interpolate, Resize, upsampling (downsampling?)
    static layer UpSampling(layer parent, const initializer_list<int> &size, string interpolation="nearest", string name=""); //Todo: Implement

    static layer Reshape(layer parent, const initializer_list<int> &shape, string name="");

    static layer Transpose(layer parent, const initializer_list<int> &dims, string name=""); //Todo: Implement

    // ---- LOSSES ----
    static loss LossFunc(string type);

    // ---- METRICS ----
    static metric MetricFunc(string type);

    // ---- MERGE LAYERS ----
    static layer Add(const initializer_list<layer> &layers, string name="");

    static layer Average(const initializer_list<layer> &layers, string name=""); //Todo: Implement

    static layer Concat(const initializer_list<layer> &layers, string name="");

    static layer MatMul(const initializer_list<layer> &layers, string name=""); //Todo: Implement

    static layer Maximum(const initializer_list<layer> &layers, string name=""); //Todo: Implement

    static layer Minimum(const initializer_list<layer> &layers, string name=""); //Todo: Implement

    static layer Subtract(const initializer_list<layer> &layers, string name=""); //Todo: Implement


    // ---- NOISE LAYERS ----
    static layer GaussianNoise(layer parent, float stddev, string name=""); //Todo: Implement


    // ---- NORMALIZATION LAYERS ----
    static layer BatchNormalization(layer parent, float momentum=0.99f, float epsilon=0.001f, bool affine=true, string name=""); //Todo: Implement

    static layer Dropout(layer parent, float rate, string name=""); //Todo: Implement

    // ---- OPERATOR LAYERS ----
    static layer Abs(layer l);
    static layer Diff(layer l1, layer l2);
    static layer Diff(layer l1, float k);
    static layer Div(layer l1, layer l2);
    static layer Div(layer l1, float k);
    static layer Exp(layer l);
    static layer Log(layer l);
    static layer Log2(layer l);
    static layer Log10(layer l);
    static layer Mean(layer l);
    static layer Mean(layer l, int axis);
    static layer Mean(layer l, bool keepdims);
    static layer Mean(layer l, int axis, bool keepdims);
    static layer Mult(layer l1, layer l2);
    static layer Mult(layer l1, float k);
    static layer Pow(layer l1, layer l2);
    static layer Pow(layer l1, float k);
    static layer Sqrt(layer l);
    static layer Sum(layer l1, layer l2);
    static layer Sum(layer l1, float k);
    static layer Var(layer l);

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
    static layer AveragePool(layer parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding="none", string name="");

    static layer GlobalMaxPool(layer parent, string name=""); //Todo: Implement

    static layer GlobalAveragePool(layer parent, string name=""); //Todo: Implement

    static layer MaxPool(layer parent, const initializer_list<int> &pool_size, string padding="none", string name="");
    static layer MaxPool(layer parent, const initializer_list<int> &pool_size, const initializer_list<int> &strides, string padding="none", string name="");


    // ---- RECURRENT LAYERS ----
    static layer RNN(layer parent, int units, int num_layers, bool use_bias=true, float dropout=.0f, bool bidirectional=false, string name="");

    static layer LSTM(layer parent, int units, int num_layers, bool use_bias=true, float dropout=.0f, bool bidirectional=false, string name="");


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
    static compserv CS_CPU(int th);
    static compserv CS_GPU(const initializer_list<int> &g);
    static compserv CS_FGPA(const initializer_list<int> &f);

    // ---- MODEL METHODS ----
    static model Model(vlayer in, vlayer out);
    static void build(model net, optimizer o, const initializer_list<Loss *> &c, const initializer_list<Metric *> &m);
    static void build(model net, optimizer o, const initializer_list<Loss *> &c, const initializer_list<Metric *> &m, CompServ *cs);
    static void build2(Net *m,  optim *o, vector<Loss *> lo, vector<Metric *> me, CompServ *cs);
    static string summary(model m);
    static void plot(model m, string fname);
    static void fit(model m, const initializer_list<LTensor *> &in, const initializer_list<LTensor *> &out, int batch, int epochs);
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

    // ---- MODELS ----
    static model get_model_mlp();
    static model get_model_cnn();
};

extern EDDL eddl;

#endif
