/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_FUSED_H
#define EDDL_LAYER_FUSED_H

#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"
#include "eddl/regularizers/regularizer.h"


#define TRMODE 1
#define TSMODE 0

using namespace std;

const int CPO = 4;
/// Conv + MaxPool Layer
class LConvMaxPool : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;
    PoolDescriptor *pd;

    // constructors and clones
    LConvMaxPool(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &conv_strides,
                        string conv_padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, 
                        const vector<int> &pool_size, const vector<int> &pool_strides, string pool_padding, bool use_bias, string name, int dev, int mem);

    LConvMaxPool(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &conv_strides,
                        string conv_padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, 
                        const vector<int> &pool_size, const vector<int> &pool_strides, const vector<int> &pool_padding, bool use_bias, string name, int dev, int mem);


    LConvMaxPool(Layer *parent, ConvolDescriptor *D, PoolDescriptor *P, string name, int dev, int mem);
 
    // Destructor
    ~LConvMaxPool();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void mem_delta() override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void initialize() override;

	void update_weights(Tensor* w, Tensor* bias=nullptr) override;

	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias=nullptr) override;

	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// Conv2D + ReLU Layer
class LConvReLU : public LinLayer {
public:
    static int total_layers;
	bool distributed_training;

    ConvolDescriptor *cd;

    LConvReLU(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
          int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem);

    LConvReLU(Layer *parent, ConvolDescriptor *cd, string name, int dev, int mem);

   // Destructor
    ~LConvReLU();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void mem_delta() override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void initialize() override;

	void update_weights(Tensor* w, Tensor* bias=nullptr) override;

	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias=nullptr) override;

	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// Conv2D + LeakyReLU Layer
class LConvLeakyReLU : public LinLayer {
public:
    static int total_layers;
	bool distributed_training;
    float alpha;

    ConvolDescriptor *cd;

    LConvLeakyReLU(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
          int groups, const vector<int> &dilation_rate, bool use_bias, float alpha, string name, int dev, int mem);

    LConvLeakyReLU(Layer *parent, ConvolDescriptor *cd, float alpha, string name, int dev, int mem);

   // Destructor
    ~LConvLeakyReLU();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void mem_delta() override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void initialize() override;

	void update_weights(Tensor* w, Tensor* bias=nullptr) override;

	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias=nullptr) override;

	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// LConvReLUMaxPool Layer
class LConvReLUMaxPool : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;
    PoolDescriptor *pd;

    // constructors and clones
    LConvReLUMaxPool(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &conv_strides,
                        string conv_padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, 
                        const vector<int> &pool_size, const vector<int> &pool_strides, string pool_padding, bool use_bias, string name, int dev, int mem);

    LConvReLUMaxPool(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &conv_strides,
                        string conv_padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, 
                        const vector<int> &pool_size, const vector<int> &pool_strides, const vector<int>  &pool_padding, bool use_bias, string name, int dev, int mem);


    LConvReLUMaxPool(Layer *parent, ConvolDescriptor *D, PoolDescriptor *P, string name, int dev, int mem);
    
    // Destructor
    ~LConvReLUMaxPool();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void mem_delta() override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void initialize() override;

	void update_weights(Tensor* w, Tensor* bias=nullptr) override;

	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias=nullptr) override;

	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// LConv2d_Relu Layer
class LConv2dActivation : public LinLayer {
public:
    static int total_layers;
    string act;
    ConvolDescriptor *cd;


    // constructors and clones
    LConv2dActivation(Layer *parent, string act, int filters, const vector<int> &kernel_size, const vector<int> &strides,
                      string padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem);

    LConv2dActivation(Layer *parent, string act, ConvolDescriptor *cd, string name, int dev, int mem);

    // Destructor
    ~LConv2dActivation();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void mem_delta() override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void initialize() override;

	void update_weights(vector<Tensor*> weights) override;

	void accumulate_accumulated_gradients(vector<Tensor*> grads) override;

	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// Conv + Softplus + Tanh + Mult Layer
class LConvSTM : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;


    // constructors and clones
    LConvSTM(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides,
                      string padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem);

    LConvSTM(Layer *parent, ConvolDescriptor *cd, string name, int dev, int mem);

    // Destructor
    ~LConvSTM();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void mem_delta() override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    void initialize() override;

	void update_weights(Tensor* w, Tensor* bias=nullptr) override;

	void accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias=nullptr) override;

	void reset_accumulated_gradients() override;

	void apply_accumulated_gradients() override;

    string plot(int c) override;

	static void reset_name_counter();

	void enable_distributed() override;

};

/// Conv + Softplus + Tanh + Mult + Add Layer
class LConvSTMAdd : public MLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;


    // constructors and clones
    LConvSTMAdd(vector<Layer *> parent, int filters, const vector<int> &kernel_size, const vector<int> &strides,
                      string padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem);

    LConvSTMAdd(vector<Layer *> parent, ConvolDescriptor *cd, string name, int dev, int mem);

    // Destructor
    ~LConvSTMAdd();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    // implementation
    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;

};


#endif //EDDL_LAYER_FUSED
