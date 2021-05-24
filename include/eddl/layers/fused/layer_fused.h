/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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


/// LConvSTM Layer
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

/// LConvMaxPool Layer
class LConvMaxPool : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;


    // constructors and clones
    LConvMaxPool(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides,
                      string padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem);

    LConvMaxPool(Layer *parent, ConvolDescriptor *cd, string name, int dev, int mem);

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


/// LConvReLUMaxPool Layer
class LConvReLUMaxPool : public LinLayer {
public:
    static int total_layers;

    ConvolDescriptor *cd;


    // constructors and clones
    LConvReLUMaxPool(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides,
                      string padding, const vector<int> &pads, int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem);

    LConvReLUMaxPool(Layer *parent, ConvolDescriptor *cd, string name, int dev, int mem);

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

#endif //EDDL_LAYER_FUSED_CONV_STM
