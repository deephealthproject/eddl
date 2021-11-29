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

#include "eddl/layers/conv/layer_conv.h"

using namespace std;


int LConvT3D::total_layers = 0;

// constructors and clones

LConvT3D::LConvT3D(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
                   int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem) :
        LConvT3D(parent, new ConvolDescriptorT3D(filters, kernel_size, strides, padding, pads, groups, dilation_rate, use_bias, mem), name, dev, mem) {
};

LConvT3D::LConvT3D(Layer *parent, ConvolDescriptorT3D *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 5) msg("LConvT only works over 5D tensors", "LConvT::LConvT3D");

    // Check dev with tensor dev

    // Set default name
    if(name.empty()) this->name = "convT3d" + to_string(++total_layers);

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

    // Check padding asymmetries
    if(D->pads[0] != D->pads[1] || D->pads[2] != D->pads[3] || D->pads[4] != D->pads[5]){
        msg("Padding asymmetry detected. (front=" + to_string(D->pads[0]) + ", back=" + to_string(D->pads[1]) + ", top=" + to_string(D->pads[2]) + ", bottom=" + to_string(D->pads[3]) + ", left=" + to_string(D->pads[4]) + ", right=" + to_string(D->pads[5]) + ").\nLayer name: " + this->name, "LConvT3D::LConvT3D");
    }
}


LConvT3D::~LConvT3D(){
    delete cd;
}

// virtual
void LConvT3D::resize(int batch){
    cd->resize(batch);
}

void LConvT3D::mem_delta(){
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

void LConvT3D::forward() {
    tensorNN::ConvT3D(this->cd);
}

void LConvT3D::backward() {
    //get gradients with provided delta
    if (trainable) { tensorNN::ConvT3D_grad(this->cd); }
    //else {cout<<name<<" not trainable"<<endl;}

    // backprop delta
    if (this->parent.size()) {
        tensorNN::ConvT3D_back(this->cd);
    }

    // Regularizer
    if (trainable) if(reg!= nullptr) {reg->apply(cd->K);}
}

void LConvT3D::initialize() {
    init->apply(params[0]);  // Conv
    params[1]->fill_(0.0f); // Bias
}

void LConvT3D::update_weights(Tensor* w, Tensor* bias) {
    Tensor::copy( w, cd->K );
    if ( bias != nullptr ) Tensor::copy( bias, cd->bias );
}

void LConvT3D::accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias) {
    cd->K->add_( gw );
    if ( gbias != nullptr ) cd->bias->add_( gbias );
}

void LConvT3D::reset_accumulated_gradients() {
    cd->acc_gK->fill_(0.0);
    cd->acc_gbias->fill_(0.0);
}

void LConvT3D::apply_accumulated_gradients() {
    cd->K->add_( cd->acc_gK );
    cd->bias->add_( cd->acc_gbias );

    // Regularizer
    if(reg!= nullptr) {reg->apply(cd->K);}
}

Layer *LConvT3D::share(int c, int bs, vector<Layer *> p) {
    LConvT3D *n = new LConvT3D(p[0], cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias, "share_" + to_string(c) + this->name, this->dev, this->mem_level);

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

Layer *LConvT3D::clone(int c, int bs, vector<Layer *> p, int todev) {
    LConvT3D *n = new LConvT3D(p[0], cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias, this->name, todev, this->mem_level);
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


string LConvT3D::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LConvT3D::reset_name_counter() {
    total_layers = 0;
}

void LConvT3D::enable_distributed() {
    distributed_training = true;
    cd->enable_distributed();

    acc_gradients.push_back(cd->acc_gK);
    acc_gradients.push_back(cd->acc_gbias);
}
