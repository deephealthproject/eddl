/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _EDDL_
#define _EDDL_

#include <vector>
#include <thread>
#include <pthread.h>

#include "../net.h"
#include "../initializers/initializer.h"
#include "../regularizers/regularizer.h"
#include "../losses/loss.h"
#include "../metrics/metric.h"

#include "../layers/layer.h"
#include "../layers/conv/layer_conv.h"
#include "../layers/core/layer_core.h"
#include "../layers/generators/layer_generators.h"
#include "../layers/merge/layer_merge.h"
#include "../layers/noise/layer_noise.h"
#include "../layers/normalization/layer_normalization.h"
#include "../layers/operators/layer_operators.h"
#include "../layers/reductions/layer_reductions.h"
#include "../layers/pool/layer_pool.h"
#include "../layers/recurrent/layer_recurrent.h"



namespace eddl {

#define layer Layer*
#define model Net*
#define optimizer Optimizer*
#define initializer Initializer*
#define regularizer Regularizer*
#define loss Loss*
#define metric Metric*
#define compserv CompServ*

// ---- CORE LAYERS ----
    layer Activation(layer parent, string activation, string name = "");
    layer Softmax(layer parent);
    layer Sigmoid(layer parent);
    layer ReLu(layer parent);


    layer Conv(layer parent, int filters, const vector<int> &kernel_size,
               const vector<int> &strides = {1, 1}, string padding = "same", int groups = 1,
               const vector<int> &dilation_rate = {1, 1},
               bool use_bias = true, string name = "");
    layer ConvT(layer parent, int filters, const vector<int> &kernel_size,
                const vector<int> &output_padding, string padding = "same",
                const vector<int> &dilation_rate = {1, 1},
                const vector<int> &strides = {1, 1}, bool use_bias = true, string name = ""); //Todo: Implement
    layer Dense(layer parent, int ndim, bool use_bias = true,  string name = "");
    layer Embedding(int input_dim, int output_dim, string name = ""); //Todo: Implement
    layer Input(const vector<int> &shape, string name = "");

    layer UpSampling(layer parent, const vector<int> &size, string interpolation = "nearest",
                     string name = ""); //Todo: Implement
    layer Reshape(layer parent, const vector<int> &shape, string name = "");

    layer Transpose(layer parent, const vector<int> &dims, string name = ""); //Todo: Implement

// ---- LOSSES ----
    loss getLoss(string type);

// ---- METRICS ----
    metric getMetric(string type);


// ---- MERGE LAYERS ----
    layer Add(const vector<layer> &layers, string name = "");

    layer Average(const vector<layer> &layers, string name = ""); //Todo: Implement
    layer Concat(const vector<layer> &layers, string name = "");

    layer MatMul(const vector<layer> &layers, string name = ""); //Todo: Implement
    layer Maximum(const vector<layer> &layers, string name = ""); //Todo: Implement
    layer Minimum(const vector<layer> &layers, string name = ""); //Todo: Implement
    layer Subtract(const vector<layer> &layers, string name = ""); //Todo: Implement

// ---- NOISE LAYERS ----
    layer GaussianNoise(layer parent, float stddev, string name = ""); //Todo: Implement

// ---- NORMALIZATION LAYERS ----
    layer BatchNormalization(layer parent, float momentum = 0.9f, float epsilon = 0.001f, bool affine = true,
                             string name = "");
    layer Norm(layer parent, float epsilon = 0.001f, string name = "");

    layer NormMax(layer parent, float epsilon = 0.001f, string name = "");

    layer NormMinMax(layer parent, float epsilon = 0.001f, string name = "");

    layer Dropout(layer parent, float rate, string name = ""); //Todo: Implement

// ---- OPERATOR LAYERS ----
    layer Abs(layer l);

    layer Diff(layer l1, layer l2);

    layer Diff(layer l1, float k);

    layer Diff(float k, layer l1);

    layer Div(layer l1, layer l2);

    layer Div(layer l1, float k);

    layer Div(float k, layer l1);

    layer Exp(layer l);

    layer Log(layer l);

    layer Log2(layer l);

    layer Log10(layer l);

    layer Mult(layer l1, layer l2);

    layer Mult(layer l1, float k);

    layer Mult(float k,layer l1);

    layer Pow(layer l1, layer l2);

    layer Pow(layer l1, float k);

    layer Sqrt(layer l);

    layer Sum(layer l1, layer l2);

    layer Sum(layer l1, float k);

    layer Sum(float k, layer l1);

// ---- REDUCTION LAYERS ----
    layer ReduceMean(layer l, vector<int> axis = {0}, bool keepdims = false);

    layer ReduceVar(layer l, vector<int> axis = {0}, bool keepdims = false);

    layer ReduceSum(layer l, vector<int> axis = {0}, bool keepdims = false);

    layer ReduceMax(layer l, vector<int> axis = {0}, bool keepdims = false);

    layer ReduceMin(layer l, vector<int> axis = {0}, bool keepdims = false);

// ---- GENERATOR LAYERS ----
    layer GaussGenerator(float mean, float stdev, vector<int> size);

    layer UniformGenerator(float low, float high, vector<int> size);

// ---- OPTIMIZERS ----
    optimizer adadelta(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement
    optimizer adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay,
                   bool amsgrad); //Todo: Implement
    optimizer adagrad(float lr, float epsilon, float weight_decay); //Todo: Implement
    optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay); //Todo: Implement
    optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay); //Todo: Implement
    optimizer rmsprop(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement
    optimizer sgd(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);


// ---- POOLING LAYERS ----
    layer AveragePool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},
                      string padding = "none", string name = "");

    layer GlobalMaxPool(layer parent, string name = ""); //Todo: Implement
    layer GlobalAveragePool(layer parent, string name = ""); //Todo: Implement
    layer MaxPool(layer parent, const vector<int> &pool_size = {2, 2}, const vector<int> &strides = {2, 2},
                  string padding = "none", string name = "");


// ---- RECURRENT LAYERS ----
    layer RNN(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f,
              bool bidirectional = false, string name = "");

    layer LSTM(layer parent, int units, int num_layers, bool use_bias = true, float dropout = .0f,
               bool bidirectional = false, string name = "");


// ---- INITIALIZERS ----

    layer GlorotNormal(layer l,int seed=1234);
    layer GlorotUniform(layer l,int seed=1234);
    layer RandomNormal(layer l, float m=0.0,float s=0.1, float seed=1234);
    layer RandomUniform(layer l, float min=0.0,float max=0.1, float seed=1234);
    layer Constant(layer l, float v=0.1);

    // ---- REGULARIZERS ----
    layer L2(layer l,float l2);
    layer L1(layer l,float l1);
    layer L1L2(layer l,float l1,float l2);

// ---- COMPUTING SERVICES ----
    compserv CS_CPU(int th=-1);
    compserv CS_GPU(const vector<int> &g,int lsb=1);
    compserv CS_FGPA(const vector<int> &f,int lsb=1);
    compserv CS_COMPSS(char* path);


// ---- FINE-GRAINED METHODS ----
    vector<int> random_indices(int batch_size, int num_samples);

    void resize_model(model net, int batch_size);

    void set_mode(model net, int mode);

    void train_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);
    void eval_batch(model net, vector<Tensor *> in, vector<Tensor *> out, vector<int> indices);


// ---- MODEL METHODS ----
    model Model(vlayer in, vlayer out);

    void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs=nullptr);

    void summary(model m);

    void load(model m, string fname);

    void save(model m, string fname);

    void plot(model m, string fname);

    void fit(model m, const vector<Tensor *> &in, const vector<Tensor *> &out, int batch, int epochs);

    void evaluate(model m, const vector<Tensor *> &in, const vector<Tensor *> &out);

    void predict(model m, const vector<Tensor *> &in, const vector<Tensor *> &out);


// ---- DATASETS ----
    bool exist(string name);
    void download_mnist();
    void download_cifar10();

}
#endif
