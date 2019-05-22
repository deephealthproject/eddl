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
#include "../callbacks.h"
#include "../initializer.h"
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

#ifdef __cplusplus
extern "C"
{
#endif


// Create Tensors
Tensor *Tensor_init(const int *shape, int shape_size);
int Tensor_device(Tensor *t);
int Tensor_ndim(Tensor *t);
int Tensor_size(Tensor *t);  // Data length
int *Tensor_shape(Tensor *t);
float *Tensor_getData(Tensor *t);
void Tensor_point2data(Tensor *t, float *ptr);


// Create Layers
tensor LTensor_init(const int *shape, int shape_size);
tensor LTensor_init_fromfile(const char *fname);
void LTensor_div(tensor t, float v);

layer Input_init(Tensor *in, const char *name);

layer Dense_init(layer parent, int ndim, bool use_bias, const char *name);

layer Conv_init(layer parent, const int *ks, int ks_size, const int *st, int st_size, const char *p, const char *name);

layer MPool_init(layer parent, const int *ks, int ks_size, const int *st, int st_size, const char *p, const char *name);

layer Activation_init(layer parent, const char *act, const char *name);

layer Reshape_init(layer parent, const int *shape, int shape_size, const char *name);

layer Drop_init(layer parent, float df, const char *name);

layer Add_init(Layer **parent, int parent_size, const char *name);

layer Cat_init(Layer **init, int init_size, const char *name);

// Create net
model Model_init(Layer *in, int in_size, Layer *out, int out_size);

// Net operations
void plot(model m, const char *fname);
void summary(model m);
void build(model net, optimizer opt, Loss **c, int size_c, Metric **m, int size_m, compserv cs);
void fit(model m, Tensor *in, Tensor *out, int batch, int epochs);
void evaluate(model m, Tensor *in, Tensor *out);

// data
//static void download_mnist();

// Layer properties
const char *Layer_name(layer l);
Tensor *Layer_input(layer l);
Tensor *Layer_output(layer l);


// Optimizers
optimizer SGD_init(float lr, float mu);


// Computing service
compserv CS_CPU_init(int th);

#ifdef __cplusplus
}
#endif


#endif //EDDLL_WRAPPER_H
