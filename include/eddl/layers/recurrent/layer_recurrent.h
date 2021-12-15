/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_LAYER_RECURRENT_H
#define EDDL_LAYER_RECURRENT_H


#include <string>
#include <cstdio>

#include "eddl/layers/layer.h"

#define TRMODE 1
#define TSMODE 0

using namespace std;


/// RNN Layer
class LCopyStates : public MLayer {
public:
    static int total_layers;

    LCopyStates(vector<Layer *> in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;
};

/// RNN Layer
class LStates : public MLayer {
public:
    static int total_layers;

    LStates(Tensor *in, string name, int dev, int mem);

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void resize(int batch) override;

    string plot(int c) override;
};

/// RNN Layer
class LRNN : public MLayer {
public:
    int units;
    bool use_bias;
    bool bidirectional;
    static int total_layers;
    string activation;
    Layer *cps;

    Tensor *preoutput;

    Tensor *Wx;
    Tensor *gWx;
    Tensor *acc_gWx;
    Tensor *bias;
    Tensor *gbias;
    Tensor *acc_gbias;

    Tensor *Wy;
    Tensor *gWy;
    Tensor *acc_gWy;
    Tensor *biasy;


    LRNN(vector<Layer *> in, int units, string activation, bool use_bias, bool bidirectional, string name, int dev, int mem);

    ~LRNN();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void forward() override;

    void backward() override;

    void update_weights(vector<Tensor*> weights) override;

    void accumulate_accumulated_gradients(vector<Tensor*> grads) override;

    void reset_accumulated_gradients() override;

    void apply_accumulated_gradients() override;

    string plot(int c) override;

    void enable_distributed() override;
};


/// LSTM Layer
class LLSTM : public MLayer {
public:
    int units;
    bool use_bias;
    bool bidirectional;
    static int total_layers;
    bool mask_zeros;

    Layer *cps;

    Tensor *state_c;
    Tensor *state_h;
    Tensor *delta_h;
    Tensor *delta_c;

    Tensor *Wih,*Wix;
    Tensor *Wfh,*Wfx;
    Tensor *Woh,*Wox;
    Tensor *Wch,*Wcx;

    Tensor *gWih,*gWix;
    Tensor *gWfh,*gWfx;
    Tensor *gWoh,*gWox;
    Tensor *gWch,*gWcx;

    Tensor *acc_gWih,*acc_gWix;
    Tensor *acc_gWfh,*acc_gWfx;
    Tensor *acc_gWoh,*acc_gWox;
    Tensor *acc_gWch,*acc_gWcx;

    Tensor *in,*fn,*on,*cn;
    Tensor *inbias,*fnbias,*onbias,*cnbias;
    Tensor *ginbias,*gfnbias,*gonbias,*gcnbias;
    Tensor *acc_ginbias,*acc_gfnbias,*acc_gonbias,*acc_gcnbias;

    Tensor *incn,*cn1fn;
    Tensor *sh;

    Tensor *mask;
    Tensor *psh;
    Tensor *psc;


    LLSTM(vector<Layer *> in, int units,  bool mask_zeros, bool bidirectional, string name, int dev, int mem);

    ~LLSTM();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;
    void mem_delta() override;
    void free_delta() override;

    void forward() override;

    void backward() override;

    void update_weights(vector<Tensor*> weights) override;

    void accumulate_accumulated_gradients(vector<Tensor*> grads) override;

    void reset_accumulated_gradients() override;

    void apply_accumulated_gradients() override;

    string plot(int c) override;

    void enable_distributed() override;
};


/// GRU Layer
class LGRU : public MLayer {
public:
    int units;
    bool use_bias;
    bool bidirectional;
    static int total_layers;
    bool mask_zeros;

    Layer *cps;

    // Output hidden state
    Tensor *state_hidden;
    Tensor *delta_hidden;

    // Weights and biases of gates: z, r, h
    Tensor *Uz_h, *Wz_x;
    Tensor *Ur_h, *Wr_x;
    Tensor *Un_h, *Wn_x;
    Tensor *bias_z_t, *bias_r_t, *bias_n_t, *bias_n_t_hidden;

    // Gradient tensors
    Tensor *gUz_h, *gWz_x;
    Tensor *gUr_h, *gWr_x;
    Tensor *gUn_h, *gWn_x;
    Tensor *g_bias_z_t, *g_bias_r_t, *g_bias_n_t, *g_bias_n_t_hidden;

    // Accumulated gradient tensors for distributed training
    Tensor *acc_gUz_h, *acc_gWz_x;
    Tensor *acc_gUr_h, *acc_gWr_x;
    Tensor *acc_gUn_h, *acc_gWn_x;
    Tensor *acc_g_bias_z_t, *acc_g_bias_r_t, *acc_g_bias_n_t, *acc_g_bias_n_t_hidden;

    // Intermediate outputs of the cell
    Tensor *z_t, *r_t, *n_t; // Gates outputs
    Tensor *n_t_hidden, *one_minus_z_t; // Gates interoperations
    //Tensor *r_t_h_t_1; // Temporary tensors for back-propagation

    // Tensors for mask_zeros
    Tensor *mask;
    Tensor *prev_hidden;


    LGRU(vector<Layer *> in, int units,  bool mask_zeros, bool bidirectional, string name, int dev, int mem);

    ~LGRU();

    Layer *share(int c, int bs, vector<Layer *> p) override;

    Layer *clone(int c, int bs, vector<Layer *> p, int todev) override;

    void resize(int batch) override;
    void mem_delta() override;
    void free_delta() override;

    void forward() override;

    void backward() override;

    void update_weights(vector<Tensor*> weights) override;

    void accumulate_accumulated_gradients(vector<Tensor*> grads) override;

    void reset_accumulated_gradients() override;

    void apply_accumulated_gradients() override;

    string plot(int c) override;

    void enable_distributed() override;
};

void reduced_abs_sum(Tensor * input, Tensor *output);

Tensor *replicate_tensor(Tensor *input,int d);

#endif //EDDL_LAYER_RECURRENT_H
