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

#include "eddl/layers/fused/layer_fused.h"

using namespace std;


int LConv2dActivation::total_layers = 0;

// constructors and clones

LConv2dActivation::LConv2dActivation(Layer *parent, string act, int filters, const vector<int> &kernel_size,
                                     const vector<int> &strides, string padding, const vector<int> &pads,
                                     int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem) :
        LConv2dActivation(parent, act, new ConvolDescriptor(filters, kernel_size, strides, padding, pads, groups, dilation_rate, use_bias, mem), name, dev, mem) {
};

LConv2dActivation::LConv2dActivation(Layer *parent, string act, ConvolDescriptor *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 4) msg("LConv2d_Relu only works over 4D tensors", "LConv2dActivation::LConv2dActivation");

    this->act=act;

    // Set default name
    if(name.empty()) this->name = "conv2d_relu_" + to_string(++total_layers);

    input = parent->output;
    cd = D;
    cd->build(input);

    output = cd->O;

    params.push_back(cd->K);
    params.push_back(cd->bias);

    gradients.push_back(cd->gK);
    gradients.push_back(cd->gbias);

    distributed_training = false;
    cd->acc_gK = nullptr;
    cd->acc_gbias = nullptr;

    parent->addchild(this);
    addparent(parent);
}


LConv2dActivation::~LConv2dActivation(){
    delete cd;  
}

// virtual
void LConv2dActivation::resize(int batch){
    cd->resize(batch);
}

void LConv2dActivation::mem_delta(){
    if(this->delta == nullptr) {
        // Reserve parent's delta
        parent[0]->mem_delta();
        cd->ID = parent[0]->delta;

        delta = Tensor::zeros(cd->O->shape, cd->O->device);
        cd->D = delta;

        if(this->verbosity_level >= 2) {
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LConv2dActivation::forward() {
    tensorNN::conv2d_activation(this->act, this->cd);
}

void LConv2dActivation::backward() {
    //get gradients with provided delta
    if (trainable) { tensorNN::Conv2D_grad(this->cd); }
    //else {cout<<name<<" not trainable"<<endl;}

    // backprop delta
    if (this->parent.size()) {
        tensorNN::Conv2D_back(this->cd);
    }

    // Regularizer
    if (trainable) if(reg!= nullptr) {reg->apply(cd->K);}
}

void LConv2dActivation::initialize() {
    init->apply(params[0]);  // Conv
    params[1]->fill_(0.0f); // Bias
}

void LConv2dActivation::update_weights(vector<Tensor*> weights) {
    if (weights.size() == 2) {
        Tensor::copy(weights[0], cd->K);
        Tensor::copy(weights[1], cd->bias);
    } else if (weights.size() == 1) {
        Tensor::copy(weights[0], cd->K);
    } else {
        cerr << "[WARNING - LConv2dActivation::update_weights] "
             << "Unexpected number of weights tensors recieved "
             << "(weights.size()=" << weights.size() << ")" << endl;
    }
}

void LConv2dActivation::accumulate_accumulated_gradients(vector<Tensor*> grads) {
    if (grads.size() == 2) {
        cd->K->add_(grads[0]);
        cd->bias->add_(grads[1]);
    } else if (grads.size() == 1) {
        cd->K->add_(grads[0]);
    } else {
        cerr << "[WARNING - LConv2dActivation::accumulate_accumulated_gradients] "
             << "Unexpected number of gradient tensors recieved "
             << "(grads.size()=" << grads.size() << ")" << endl;
    }
}

void LConv2dActivation::reset_accumulated_gradients() {
    cd->acc_gK->fill_(0.0);
    cd->acc_gbias->fill_(0.0);
}

void LConv2dActivation::apply_accumulated_gradients() {
    cd->K->add_( cd->acc_gK );
    cd->bias->add_( cd->acc_gbias );

    // Regularizer
    if(reg!= nullptr) {reg->apply(cd->K);}
}

Layer *LConv2dActivation::share(int c, int bs, vector<Layer *> p) {
    LConv2dActivation *n = new LConv2dActivation(p[0], this->act, cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias, "share_" + to_string(c) + this->name, this->dev, this->mem_level);

    n->orig = this;
    n->isshared=true;
    n->trainable = trainable;
    n->do_deletes = false;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();


    n->cd->K = cd->K;
    n->cd->bias = cd->bias;

    n->params.push_back(n->cd->K);
    n->params.push_back(n->cd->bias);

    //share gradients
    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    n->cd->gK = cd->gK;
    n->cd->gbias = cd->gbias;

    n->gradients.push_back(n->cd->gK);
    n->gradients.push_back(n->cd->gbias);


    if ( distributed_training ) {
        n->acc_gradients.clear();

        n->cd->acc_gK = cd->acc_gK;
        n->cd->acc_gbias = cd->acc_gbias;

        n->acc_gradients.push_back(n->cd->acc_gK);
        n->acc_gradients.push_back(n->cd->acc_gbias);
    }

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    return n;
}

Layer *LConv2dActivation::clone(int c, int bs, vector<Layer *> p, int todev) {
    LConv2dActivation *n = new LConv2dActivation(p[0], this->act, cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias, this->name, todev, this->mem_level);
    n->trainable = trainable;
    n->do_deletes = false;

    n->orig = this;

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    if (distributed_training)
        n->enable_distributed();

    return n;
}


string LConv2dActivation::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LConv2dActivation::reset_name_counter() {
    total_layers = 0;
}

void LConv2dActivation::enable_distributed() {
    distributed_training = true;
    cd->enable_distributed();

    acc_gradients.push_back(cd->acc_gK);
    acc_gradients.push_back(cd->acc_gbias);
}
