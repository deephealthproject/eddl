/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/conv/layer_conv.h"

using namespace std;


int LConv3D::total_layers = 0;

// constructors and clones

LConv3D::LConv3D(Layer *parent, const vector<int> &ks, const vector<int> &st,
             const vector<int> &p, string name, int dev, int mem) : LConv3D(parent, new ConvolDescriptor3D(ks, st, p, mem), name, dev, mem) {}

LConv3D::LConv3D(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding,
             int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem) : LConv3D(parent, new ConvolDescriptor3D(filters, kernel_size, strides, padding, use_bias, mem), name, dev, mem) {
    // TODO: Implement (Fix initialization)
};

LConv3D::LConv3D(Layer *parent, ConvolDescriptor3D *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 5) msg("LConv3D only works over 5D tensors", "LConv3D::LConv3D");

    // Check dev with tensor dev

    // Set default name
    if(name.empty()) this->name = "conv3d" + to_string(++total_layers);

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


LConv3D::~LConv3D(){
    delete cd;  
}

// virtual
void LConv3D::resize(int batch){
    cd->resize(batch);

}

void LConv3D::initialize() {
    init->apply(params[0]);  // Conv
    params[1]->fill_(0.0f); // Bias
}

void LConv3D::mem_delta(){
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

void LConv3D::forward() {
    tensorNN::Conv3D(this->cd);
}

void LConv3D::backward() {
    //get gradients with provided delta
    if (trainable) { tensorNN::Conv3D_grad(this->cd); }

    // backprop delta
    if (this->parent.size()) {
        tensorNN::Conv3D_back(this->cd);
    }

    // Regularizer
    if (trainable) if(reg!= nullptr) {reg->apply(cd->K);}
}

void LConv3D::update_weights(Tensor* w, Tensor* bias) {
    Tensor::copy( w, cd->K );
    if ( bias != nullptr ) Tensor::copy( bias, cd->bias );
}

void LConv3D::accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias) {
    cd->K->add_( gw );
    if ( gbias != nullptr ) cd->bias->add_( gbias );
}

void LConv3D::reset_accumulated_gradients() {
    cd->acc_gK->fill_(0.0);
    cd->acc_gbias->fill_(0.0);
}

void LConv3D::apply_accumulated_gradients() {
    cd->K->add_( cd->acc_gK );
    cd->bias->add_( cd->acc_gbias );

    // Regularizer
    if(reg!= nullptr) {reg->apply(cd->K);}
}

Layer *LConv3D::share(int c, int bs, vector<Layer *> p) {
    LConv3D *n = new LConv3D(p[0], cd->ksize, cd->stride, cd->pad,  "share_"+to_string(c)+this->name, dev,mem_level);
    n->orig = this;
    n->isshared=true;
    n->trainable = trainable;
    n->do_deletes = false;

    n->cd->use_bias=cd->use_bias;

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

Layer *LConv3D::clone(int c, int bs, vector<Layer *> p, int todev) {

    LConv3D *n = new LConv3D(p[0], cd->ksize, cd->stride, cd->pad,  name, todev, this->mem_level);
    n->trainable = trainable;
    n->do_deletes = false;

    n->orig = this;
    n->cd->use_bias = cd->use_bias;

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    if (distributed_training)
        n->enable_distributed();

    return n;
}


string LConv3D::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LConv3D::reset_name_counter() {
    total_layers = 0;
}

void LConv3D::enable_distributed() {
    distributed_training = true;
    cd->enable_distributed();

    acc_gradients.push_back(cd->acc_gK);
    acc_gradients.push_back(cd->acc_gbias);
}
