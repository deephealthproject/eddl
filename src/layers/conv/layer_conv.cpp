/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/conv/layer_conv.h"

using namespace std;


int LConv::total_layers = 0;

// constructors and clones

LConv::LConv(Layer *parent, const vector<int> &ks, const vector<int> &st,
             const vector<int> &p, string name, int dev, int mem) : LConv(parent, new ConvolDescriptor(ks, st, p, mem), name, dev, mem) {}

LConv::LConv(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding,
             int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem) : LConv(parent, new ConvolDescriptor(filters, kernel_size, strides, padding, use_bias, mem), name, dev, mem) {
    // TODO: Implement (Fix initialization)
};

LConv::LConv(Layer *parent, ConvolDescriptor *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 4) msg("LConv only works over 4D tensors", "LConv::LConv");

    // Check dev with tensor dev

    // Set default name
    if(name.empty()) this->name = "conv" + to_string(++total_layers);

    input = parent->output;
    cd = D;
    cd->build(input);

    output = cd->O;
//    delta = cd->D;
//    cd->ID = parent->delta;

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


// virtual
void LConv::resize(int batch){
    cd->resize(batch);

}

void LConv::mem_delta(){
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

void LConv::forward() {
    Conv2D(this->cd);
}

void LConv::backward() {
    //get gradients with provided delta
    if (trainable) { Conv2D_grad(this->cd); }

    // backprop delta
    if (this->parent.size()) {
        Conv2D_back(this->cd);
    }

    // Regularizer
    if (trainable) if(reg!= nullptr) {reg->apply(cd->K);}
}

void LConv::update_weights(Tensor* w, Tensor* bias) {
    Tensor::copy( w, cd->K );
    if ( bias != nullptr ) Tensor::copy( bias, cd->bias );
}

void LConv::accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias) {
    cd->K->add_( gw );
    if ( gbias != nullptr ) cd->bias->add_( gbias );
}

void LConv::reset_accumulated_gradients() {
    cd->acc_gK->fill_(0.0);
    cd->acc_gbias->fill_(0.0);
}

void LConv::apply_accumulated_gradients() {
    cd->K->add_( cd->acc_gK );
    cd->bias->add_( cd->acc_gbias );

    // Regularizer
    if(reg!= nullptr) {reg->apply(cd->K);}
}

Layer *LConv::share(int c, int bs, vector<Layer *> p) {
    LConv *n = new LConv(p[0], cd->ksize, cd->stride, cd->pad, "share_" + to_string(c) + name, dev,mem_level);
    n->orig = this;
    n->isshared=true;
    
    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();
    n->acc_gradients.clear();
    n->cd->use_bias=cd->use_bias;

    n->cd->K = cd->K;
    n->cd->bias = cd->bias;
    new(&n->cd->matK) Eigen::Map<Eigen::MatrixXf>(n->cd->K->ptr, cd->kr * cd->kc * cd->kz, cd->nk);

    n->params.push_back(n->cd->K);
    n->params.push_back(n->cd->bias);


    if ( distributed_training ) {
        n->cd->acc_gK  = cd->acc_gK;
        n->cd->acc_gbias  = cd->acc_gbias;

        n->acc_gradients.push_back(n->cd->acc_gK);
        n->acc_gradients.push_back(n->cd->acc_gbias);
    }

    n->reg=reg;
    n->init=init;

    return n;
}

Layer *LConv::clone(int c, int bs, vector<Layer *> p, int todev) {
    LConv *n = new LConv(p[0], cd->ksize, cd->stride, cd->pad, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;
    n->cd->use_bias=cd->use_bias;

    n->reg=reg;
    n->init=init;


    return n;
}


string LConv::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LConv::reset_name_counter() {
    total_layers = 0;
}

void LConv::enable_distributed() {
    distributed_training = true;
    cd->enable_distributed();

    acc_gradients.push_back(cd->acc_gK);
    acc_gradients.push_back(cd->acc_gbias);
}
