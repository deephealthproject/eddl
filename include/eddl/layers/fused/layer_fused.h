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


#endif //EDDL_LAYER_FUSED_H
