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


int LConv1D::total_layers = 0;

// constructors and clones

LConv1D::LConv1D(Layer *parent, int filters, const vector<int> &kernel_size, const vector<int> &strides, string padding, const vector<int> &pads,
             int groups, const vector<int> &dilation_rate, bool use_bias, string name, int dev, int mem) : LConv1D(parent, new ConvolDescriptor(filters, kernel_size, strides, padding, pads, groups, dilation_rate, use_bias, mem), name, dev, mem) {
};

LConv1D::LConv1D(Layer *parent, ConvolDescriptor *D, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    if (parent->output->ndim != 3) msg("LConv only works over 3D tensors", "LConv1D::LConv1D");

    // Check dev with tensor dev

    // Set default name
    if(name.empty()) this->name = "conv1d" + to_string(++total_layers);

    input = parent->output;

    // Reshape the 2D input to a 3D tensor
    vector<int> in_shape = input->getShape();
    in_shape.push_back(1);
    input_reshaped = new Tensor(in_shape, input);

    cd = D;
    cd->build(input_reshaped);  // Using the 3D tensor

    // Reshape the 3D output from conv to a 2D tensor
    vector<int> out_shape = cd->O->getShape();
    out_shape.pop_back();
    output = new Tensor(out_shape, cd->O);

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


LConv1D::~LConv1D(){
    delete input_reshaped;
    input_reshaped = nullptr;

    // deleting cd->O later in this method can drive to double delete/free 
    Tensor *O_temp = cd->O;

    // deleting cd->D here can drive to double delete/free 
    if (cd->D != nullptr) delete cd->D;
    cd->D = nullptr;

    delete cd;
    cd = nullptr;

    // TODO check where is the proper place to delete/free cd->O
    if (O_temp != nullptr) delete O_temp;
}

// virtual
void LConv1D::resize(int batch){
    // Resize but keeping the pointer to the input before the reshape
    input_reshaped->resize(batch, input->ptr); 

    cd->resize(batch);

    // Resize but keeping the pointer to the output of the descriptor
    output->resize(batch, cd->O->ptr);
}

void LConv1D::initialize() {
    init->apply(params[0]);  // Conv
    params[1]->fill_(0.0f); // Bias
}

void LConv1D::mem_delta(){
    if(this->delta == nullptr) {
        // Reserve parent's delta
        parent[0]->mem_delta();
        cd->ID = parent[0]->delta;

        // Show delta with the output shape of the Conv1D
        delta = Tensor::zeros(output->shape, output->device);
        // Reshape delta for convol descriptor
        if (cd->D != nullptr) delete cd->D;
        cd->D = new Tensor(cd->O->shape, delta);

        if(this->verbosity_level >= 2) {
            std::cout << "Booked delta for: " + this->name << std::endl;
        }
    }
}

void LConv1D::forward() {
    tensorNN::Conv2D(this->cd);
}

void LConv1D::backward() {
    //get gradients with provided delta
    if (trainable) { tensorNN::Conv2D_grad(this->cd); }

    // backprop delta
    if (this->parent.size()) {
        tensorNN::Conv2D_back(this->cd);
    }

    // Regularizer
    if (trainable) if(reg!= nullptr) {reg->apply(cd->K);}
}

void LConv1D::update_weights(vector<Tensor*> weights) {
    if (weights.size() == 2) {
        Tensor::copy(weights[0], cd->K);
        Tensor::copy(weights[1], cd->bias);
    } else if (weights.size() == 1) {
        Tensor::copy(weights[0], cd->K);
    } else {
        cerr << "[WARNING - LConv1D::update_weights] "
             << "Unexpected number of weights tensors recieved "
             << "(weights.size()=" << weights.size() << ")" << endl;
    }
}

void LConv1D::accumulate_accumulated_gradients(vector<Tensor*> grads) {
    if (grads.size() == 2) {
        cd->K->add_(grads[0]);
        cd->bias->add_(grads[1]);
    } else if (grads.size() == 1) {
        cd->K->add_(grads[0]);
    } else {
        cerr << "[WARNING - LConv1D::accumulate_accumulated_gradients] "
             << "Unexpected number of gradient tensors recieved "
             << "(grads.size()=" << grads.size() << ")" << endl;
    }
}

void LConv1D::reset_accumulated_gradients() {
    cd->acc_gK->fill_(0.0);
    cd->acc_gbias->fill_(0.0);
}

void LConv1D::apply_accumulated_gradients() {
    cd->K->add_( cd->acc_gK );
    cd->bias->add_( cd->acc_gbias );

    // Regularizer
    if(reg!= nullptr) {reg->apply(cd->K);}
}

Layer *LConv1D::share(int c, int bs, vector<Layer *> p) {
    LConv1D *n = new LConv1D(p[0], cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias,  "share_"+to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared = true;
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

        n->cd->acc_gK  = cd->acc_gK;
        n->cd->acc_gbias  = cd->acc_gbias;

        n->acc_gradients.push_back(n->cd->acc_gK);
        n->acc_gradients.push_back(n->cd->acc_gbias);
    }
    
    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    return n;
}

Layer *LConv1D::clone(int c, int bs, vector<Layer *> p, int todev) {

    LConv1D *n = new LConv1D(p[0], cd->filters, cd->kernel_size, cd->strides, cd->padding, cd->pads, cd->groups, cd->dilation_rate, cd->use_bias,  this->name, todev, this->mem_level);
    n->trainable = trainable;
    n->do_deletes = false;

    n->orig = this;

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;


    return n;
}


string LConv1D::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=gray,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}

void LConv1D::reset_name_counter() {
    total_layers = 0;
}

void LConv1D::enable_distributed() {
    distributed_training = true;
    cd->enable_distributed();

    acc_gradients.push_back(cd->acc_gK);
    acc_gradients.push_back(cd->acc_gbias);
}
