/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/core/layer_core.h"

using namespace std;

int LDense::total_layers = 0;

LDense::LDense(Layer *parent, int ndim, bool use_bias, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 2) msg("LDense only works over 2D tensors", "LDense");

    if(name.empty()) this->name = "dense" + to_string(++total_layers);
    this->ndim = ndim;
    this->use_bias = use_bias;

    input = parent->output;
    output = new Tensor(vector<int>{input->shape[0], ndim}, dev);


    W = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    if (use_bias) bias = new Tensor(vector<int>{ndim}, dev);
    params.push_back(W);
    if (use_bias) params.push_back(bias);

    gW = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    if (use_bias) gbias = new Tensor(vector<int>{ndim}, dev);
    gradients.push_back(gW);
    if (use_bias) gradients.push_back(gbias);

    distributed_training = false;
    acc_gW = nullptr;
    acc_gbias = nullptr;

    parent->addchild(this);
    addparent(parent);
}

LDense::~LDense(){
    // input, output, delta, params[], and gradients[], acc_gradients[] => deleted in ~Layer()
}

void LDense::forward() {
    if (quantization_mode > 0) {
        if (qW==nullptr) qW = new Tensor(W->getShape(), W->device);
        if (qbias==nullptr) qbias = new Tensor(bias->getShape(), bias->device);

        Tensor::copy(W, qW);
        W->quantize_(quantization_clipping_bits, quantization_rounding_bits, 1);

        Tensor::copy(bias, qbias);
        bias->quantize_(quantization_clipping_bits, quantization_rounding_bits, 1);

        if(quantization_mode < 3) input->quantize_(quantization_clipping_bits, quantization_rounding_bits,1);

    }

    Tensor::mult2D(input, 0, W, 0, output, 0);
    if(quantization_mode >0) {
        if(quantization_mode<3)output->quantize_(quantization_clipping_bits, quantization_rounding_bits, 1);
        else  output->clipping_(std::pow(2,quantization_clipping_bits)/2-1,(std::pow(2,quantization_clipping_bits)/2-1)*-1 );
    }

    if (use_bias) Tensor::sum2D_rowwise(output, bias, output);

    if(quantization_mode > 0) {
        if(quantization_mode<3)output->quantize_(quantization_clipping_bits, quantization_rounding_bits, 1);
        else  output->clipping_(std::pow(2,quantization_clipping_bits)/2-1,(std::pow(2,quantization_clipping_bits)/2-1)*-1 );
        Tensor::copy(qW, W);
        Tensor::copy(qbias, bias);
    }
    
    float max_h = Tensor::max(output);
    if (max_h > max_quant) max_quant = max_h;
    float min_h = Tensor::min(output);
    if (min_h < min_quant) min_quant = min_h;

     max_h = Tensor::max(input);
    if (max_h > max_quant) max_quant = max_h;
     min_h = Tensor::min(input);
    if (min_h < min_quant) min_quant = min_h;

     max_h = Tensor::max( W);
    if (max_h > max_quant) max_quant = max_h;
     min_h = Tensor::min( W);
    if (min_h < min_quant) min_quant = min_h;

     max_h = Tensor::max(bias);
    if (max_h > max_quant) max_quant = max_h;
     min_h = Tensor::min(bias);
    if (min_h < min_quant) min_quant = min_h;
}

void LDense::backward() {
    //get gradients with provided delta
    if (trainable) {
        Tensor::mult2D(input, 1, delta, 0, gW, 1);
        if (use_bias) Tensor::reduce_sum2D(delta, gbias, 0, 1);
    }//else {cout<<name<<" not trainable"<<endl;}

    //1: note that increment parent delta
    Tensor::mult2D(delta, 0, W, 1, parent[0]->delta, 1);

    // Regularizer
    if (trainable) if(reg != nullptr) {reg->apply(this->W);}
}

void LDense::update_weights(vector<Tensor*> weights) {
    if (weights.size() == 2) {
        Tensor::copy(weights[0], this->W);
        Tensor::copy(weights[1], this->bias);
    } else if (weights.size() == 1) {
        Tensor::copy(weights[0], this->W);
    } else {
        cerr << "[WARNING - LDense::update_weights] "
             << "Unexpected number of weights tensors recieved "
             << "(weights.size()=" << weights.size() << ")" << endl;
    }
}

void LDense::accumulate_accumulated_gradients(vector<Tensor*> grads) {
    if (grads.size() == 2) {
        W->add_(grads[0]);
        bias->add_(grads[1]);
    } else if (grads.size() == 1) {
        W->add_(grads[0]);
    } else {
        cerr << "[WARNING - LDense::accumulate_accumulated_gradients] "
             << "Unexpected number of gradient tensors recieved "
             << "(grads.size()=" << grads.size() << ")" << endl;
    }

    // Regularizer
    if(reg != nullptr) { reg->apply(this->W); }
}

void LDense::reset_accumulated_gradients() {
    acc_gW->fill_(0.0);
    if (use_bias) acc_gbias->fill_(0.0);
}

void LDense::apply_accumulated_gradients() {
    W->add_( acc_gW );
    if ( use_bias ) bias->add_( acc_gbias );

    // Regularizer
    if(reg != nullptr) { reg->apply(this->W); }
}


Layer *LDense::share(int c, int bs, vector<Layer *> p) {
    LDense *n = new LDense(p[0], ndim, use_bias, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared = true;
    n->trainable = trainable;
    n->do_deletes = false;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->W = params[0];
    if (use_bias) n->bias = params[1];

    if ( distributed_training ) {
        n->acc_gradients.clear();

        n->acc_gW = this->acc_gradients[0];
        n->acc_gradients.push_back(n->acc_gW);
        if ( use_bias ) {
            n->acc_gbias = this->acc_gradients[1];
            n->acc_gradients.push_back(n->acc_gbias);
        }
    }

    n->params.push_back(n->W);
    if (use_bias) n->params.push_back(n->bias);

    //share gradients
    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    n->gW = gradients[0];
    if (use_bias) n->gbias = gradients[1];

    n->gradients.push_back(n->gW);
    if (use_bias) n->gradients.push_back(n->gbias);

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    return n;
}

Layer *LDense::clone(int c, int bs, vector<Layer *> p, int todev) {
    LDense *n = new LDense(p[0], ndim, use_bias,  "clone_" + name, todev, this->mem_level);
    n->orig = this;
    n->trainable = trainable;
    n->do_deletes = false;

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    if (distributed_training)
        n->enable_distributed();

    return n;
}


string LDense::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LDense::reset_name_counter(){
    total_layers=0;
}

void LDense::enable_distributed(){
    distributed_training = true;

    // Tensors with the accumulation of the gradients
    acc_gW = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    acc_gW->fill_(0.0);
    acc_gradients.push_back(acc_gW);

    if (use_bias) {
        acc_gbias = new Tensor(vector<int>{ndim}, dev);
        acc_gbias->fill_(0.0);
        acc_gradients.push_back(acc_gbias);
    }
}
