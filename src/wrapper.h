//
// Created by Salva Carrión on 2019-04-10.
//

#ifndef EDDLL_WRAPPER_H
#define EDDLL_WRAPPER_H


#include <initializer_list>
#include <vector>
#include "net.h"

#define tensor LTensor*
#define layer Layer*
#define model Net*

#ifdef __cplusplus
extern "C"
{
#endif


// Create Tensors
tensor T_init(const int* init, int init_size, int dev);

// Create Layers
layer Input_init(tensor t, int dev);

layer Dense_init(layer parent, int dim, const char* name, int dev);

layer Conv_init(layer parent, const int* ks, int ks_size, const int* st, int st_size, const char* p, int dev);

layer Activation_init(layer parent, const char* act, const char* name, int dev);

layer Reshape_init(layer parent, const int* init, int init_size, const char* name, int dev);

layer Drop_init(layer parent, float df, const char* name, int dev);

layer Add_init(Layer** parent, int parent_size, const char* name, int dev);

layer Cat_init(Layer** init, int init_size, const char* name, int dev);

// Create net
model Model_init(Layer* in, int in_size, Layer* out, int out_size);

// Net operations
void plot(model m, const char* fname);
void info(model m);
//void build(model net,optim *opt,const initializer_list<string>& c, const initializer_list<string>& m, int todev);
//void fit(model m, const initializer_list<LTensor*>& in, const initializer_list<LTensor*>& out, int batch, int epochs);
//void evaluate(model m, const initializer_list<LTensor*>& in, const initializer_list<LTensor*>& out);
//
//
//
//// data
//static void download_mnist();

const char* Layer_name(layer l);



#ifdef __cplusplus
}
#endif


#endif //EDDLL_WRAPPER_H
