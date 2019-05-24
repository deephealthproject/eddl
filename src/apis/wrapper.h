//
// Created by Salva Carrión on 2019-04-10.
//

#ifndef EDDLL_WRAPPER_H
#define EDDLL_WRAPPER_H


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

#define layer Layer*
#define model Net*
#define optimizer optim*
#define callback Callback*
#define initializer Initializer*
#define loss Loss*
#define metric Metric*
#define compserv CompServ*

typedef vector<LTensor *> vltensor;

#ifdef __cplusplus
extern "C"
{
#endif


// Tensors ==============================
Tensor *Tensor_init(const int *shape, int shape_size);
int Tensor_device(Tensor *t);
int Tensor_ndim(Tensor *t);
int Tensor_size(Tensor *t);  // Data length
int *Tensor_shape(Tensor *t);
float *Tensor_getData(Tensor *t);
void Tensor_point2data(Tensor *t, float *ptr);
void Tensor_floodData(Tensor *t, float *ptr);


// Initializers =========================


// Layers ===============================
// Base layer -----------------
const char *Layer_name(layer l);
Tensor *Layer_input(layer l);
Tensor *Layer_output(layer l);

// Convolutional layers -----------------
layer Conv_init(layer parent, const int *ks, int ks_size, const int *st, int st_size, const char *p, const char *name);


// Core layers --------------------------
layer Activation_init(layer parent, const char *act, const char *name);
layer Dense_init(layer parent, int ndim, bool use_bias, const char *name);
layer Drop_init(layer parent, float df, const char *name);
layer Input_init(Tensor *in, const char *name);
layer Reshape_init(layer parent, const int *shape, int shape_size, const char *name);

layer LTensor_init(const int *shape, int shape_size);
layer LTensor_init_fromfile(const char *fname);


// Merge layers -------------------------
layer Add_init(Layer **parent, int parent_size, const char *name);
layer Concat_init(Layer **init, int init_size, const char *name);


// Noise layers -------------------------


// Operator layers ----------------------
void LTensor_div(layer t, float v);


// Pool layers --------------------------
layer MaxPool_init(layer parent, const int *ks, int ks_size, const int *st, int st_size, const char *p, const char *name);


// Recurrent layers ---------------------

// Losses ================================
loss Loss_MeanSquaredError_init();
loss Loss_CrossEntropy_init();
loss Loss_SoftCrossEntropy_init();

// Metrics ===============================
metric Metric_MeanSquaredError_init();
metric Metric_CategoricalAccuracy_init();

// Optimizers ============================
optimizer SGD_init(float lr, float mu);

// Computing Service =====================
compserv CS_CPU_init(int th);

// Net ===================================
model Model_init(Layer *in, int in_size, Layer *out, int out_size);
void plot(model m, const char *fname);
void summary(model m);
void build(model net, optimizer opt, Loss **c, int size_c, Metric **m, int size_m, compserv cs);
void fit(model m, Tensor *in, Tensor *out, int batch, int epochs);
void fit_safe(model m, const char *in, const char *out, int batch, int epochs);
void evaluate(model m, Tensor *in, Tensor *out);


#ifdef __cplusplus
}
#endif


#endif //EDDLL_WRAPPER_H
