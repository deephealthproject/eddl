
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#ifndef _EDDL_
#define _EDDL_

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
#include "../layers/generators/layer_generators.h"
#include "../layers/merge/layer_merge.h"
#include "../layers/noise/layer_noise.h"
#include "../layers/operators/layer_operators.h"
#include "../layers/reductions/layer_reductions.h"
#include "../layers/pool/layer_pool.h"
#include "../layers/recurrent/layer_recurrent.h"

#define tensor LTensor*
#define layer Layer*
#define model Net*
#define optimizer Optimizer*
#define callback Callback*
#define initializer Initializer*
#define loss Loss*
#define metric Metric*
#define compserv CompServ*

typedef vector<LTensor *> vltensor;

// ---- TENSOR ----
tensor T(const vector<int> &shape);//*
tensor T(const vector<int> &shape, float *ptr);
tensor T_load(string fname);
float * T_getptr(tensor T);

// ---- TENSOR OPERATIONS ----
void div(tensor t, float v);

// ---- CORE LAYERS ----
layer Activation(layer parent, string activation, string name="");

layer Conv(layer parent, int filters, const vector<int> &kernel_size,
                  const vector<int> &strides={1,1}, string padding="same", int groups=1, const vector<int> &dilation_rate={1,1},
                  bool use_bias=true, string name=""); //Todo: Implement

layer ConvT(layer parent, int filters, const vector<int> &kernel_size,
                   const vector<int> &output_padding, string padding="same", const vector<int> &dilation_rate={1,1},
                   const vector<int> &strides={1,1}, bool use_bias=true, string name=""); //Todo: Implement

layer Dense(layer parent, int ndim, bool use_bias=true, string name=""); //Todo: Implement

layer Embedding(int input_dim, int output_dim, string name=""); //Todo: Implement

layer Input(const vector<int> &shape, string name="");

// TODO: Interpolate, Resize, upsampling (downsampling?)
layer UpSampling(layer parent, const vector<int> &size, string interpolation="nearest", string name=""); //Todo: Implement

layer Reshape(layer parent, const vector<int> &shape, string name="");

layer Transpose(layer parent, const vector<int> &dims, string name=""); //Todo: Implement

// ---- LOSSES ----
loss LossFunc(string type);

// ---- METRICS ----
metric MetricFunc(string type);

// ---- MERGE LAYERS ----
layer Add(const vector<layer> &layers, string name="");

layer Average(const vector<layer> &layers, string name=""); //Todo: Implement

layer Concat(const vector<layer> &layers, string name="");

layer MatMul(const vector<layer> &layers, string name=""); //Todo: Implement

layer Maximum(const vector<layer> &layers, string name=""); //Todo: Implement

layer Minimum(const vector<layer> &layers, string name=""); //Todo: Implement

layer Subtract(const vector<layer> &layers, string name=""); //Todo: Implement


// ---- NOISE LAYERS ----
layer GaussianNoise(layer parent, float stddev, string name=""); //Todo: Implement


// ---- NORMALIZATION LAYERS ----
layer BatchNormalization(layer parent, float momentum=0.99f, float epsilon=0.001f, bool affine=true, string name=""); //Todo: Implement

layer Dropout(layer parent, float rate, string name=""); //Todo: Implement

// ---- OPERATOR LAYERS ----
layer Abs(layer l);
layer Diff(layer l1, layer l2);
layer Diff(layer l1, float k);
layer Div(layer l1, layer l2);
layer Div(layer l1, float k);
layer Exp(layer l);
layer Log(layer l);
layer Log2(layer l);
layer Log10(layer l);
layer Mult(layer l1, layer l2);
layer Mult(layer l1, float k);
layer Pow(layer l1, layer l2);
layer Pow(layer l1, float k);
layer Sqrt(layer l);
layer Sum(layer l1, layer l2);
layer Sum(layer l1, float k);

// ---- REDUCTION LAYERS ----
layer ReduceMean(layer l);
layer ReduceMean(layer l, vector<int> axis);
layer ReduceMean(layer l, bool keepdims);
layer ReduceMean(layer l, vector<int> axis, bool keepdims);
layer ReduceVar(layer l);
layer ReduceVar(layer l, vector<int> axis);
layer ReduceVar(layer l, bool keepdims);
layer ReduceVar(layer l, vector<int> axis, bool keepdims);
layer ReduceSum(layer l);
layer ReduceSum(layer l, vector<int> axis);
layer ReduceSum(layer l, bool keepdims);
layer ReduceSum(layer l, vector<int> axis, bool keepdims);
layer ReduceMax(layer l);
layer ReduceMax(layer l, vector<int> axis);
layer ReduceMax(layer l, bool keepdims);
layer ReduceMax(layer l, vector<int> axis, bool keepdims);
layer ReduceMin(layer l);
layer ReduceMin(layer l, vector<int> axis);
layer ReduceMin(layer l, bool keepdims);
layer ReduceMin(layer l, vector<int> axis, bool keepdims);

// ---- GENERATOR LAYERS ----
layer GaussGenerator(float mean, float stdev, vector<int> size);
layer UniformGenerator(float low, float high, vector<int> size);

// ---- OPTIMIZERS ----
optimizer adadelta(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement
optimizer adam(float lr, float beta_1, float beta_2, float epsilon, float weight_decay, bool amsgrad); //Todo: Implement
optimizer adagrad(float lr, float epsilon, float weight_decay); //Todo: Implement
optimizer adamax(float lr, float beta_1, float beta_2, float epsilon, float weight_decay); //Todo: Implement
optimizer nadam(float lr, float beta_1, float beta_2, float epsilon, float schedule_decay); //Todo: Implement
optimizer rmsprop(float lr, float rho, float epsilon, float weight_decay); //Todo: Implement
optimizer sgd(float lr=0.01f, float momentum=0.0f, float weight_decay=0.0f, bool nesterov=false);

void change(optimizer o, vector<float> &params);


// ---- POOLING LAYERS ----
layer AveragePool(layer parent, const vector<int> &pool_size);
layer AveragePool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding="none", string name="");

layer GlobalMaxPool(layer parent, string name=""); //Todo: Implement

layer GlobalAveragePool(layer parent, string name=""); //Todo: Implement

layer MaxPool(layer parent, const vector<int> &pool_size, string padding="none", string name="");
layer MaxPool(layer parent, const vector<int> &pool_size, const vector<int> &strides, string padding="none", string name="");


// ---- RECURRENT LAYERS ----
layer RNN(layer parent, int units, int num_layers, bool use_bias=true, float dropout=.0f, bool bidirectional=false, string name="");

layer LSTM(layer parent, int units, int num_layers, bool use_bias=true, float dropout=.0f, bool bidirectional=false, string name="");


//    // ---- LR SCHEDULERS ----
//    callback CosineAnnealingLR(int T_max, float eta_min, int last_epoch); //Todo: Implement
//    callback ExponentialLR(float gamma, int last_epoch); //Todo: Implement
//    callback MultiStepLR(const vector<int> &milestones, float gamma, int last_epoch); //Todo: Implement
//    callback ReduceLROnPlateau(string metric, string mode, float factor, int patience, float threshold, string threshold_mode, int cooldown, float min_lr, float eps); //Todo: Implement
//    callback StepLR(int step_size, float gamma, int last_epoch); //Todo: Implement

// ---- INITIALIZERS ----
initializer Constant(float value); //Todo: Implement
initializer Identity(float gain); //Todo: Implement
initializer GlorotNormal(float seed); //Todo: Implement
initializer GlorotUniform(float seed); //Todo: Implement
initializer RandomNormal(float mean, float stdev, int seed); //Todo: Implement
initializer RandomUniform(float minval, float maxval, int seed); //Todo: Implement
initializer Orthogonal(float gain, int seed); //Todo: Implement


// ---- COMPUTING SERVICES ----
compserv CS_CPU(int th);
compserv CS_GPU(const vector<int> &g);
compserv CS_FGPA(const vector<int> &f);

// ---- MODEL METHODS ----
model Model(vlayer in, vlayer out);

void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me);
void build(model net, optimizer o, const vector<string> &lo, const vector<string> &me, CompServ *cs);

string summary(model m);
void load(model m,string fname);
void save(model m,string fname);
void plot(model m, string fname);
void fit(model m, const vector<LTensor *> &in, const vector<LTensor *> &out, int batch, int epochs);
void evaluate(model m, const vector<LTensor *> &in, const vector<LTensor *> &out);
void predict(model m, const vector<LTensor *> &in, const vector<LTensor *> &out);
model load_model(string fname); //Todo: Implement
void save_model(model m, string fname); //Todo: Implement
void set_trainable(model m); //Todo: Implement
model zoo_models(string model_name); //Todo: Implement

// ---- LAYER METHODS ----
void set_trainable(layer l); //Todo: Implement
layer get_layer(model m, string layer_name); //Todo: Implement


// ---- DATASETS ----
void download_mnist();

// ---- MODELS ----
model get_model_mlp(int batch_size);
model get_model_cnn(int batch_size);


#endif
