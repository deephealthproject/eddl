/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/recurrent/layer_recurrent.h"


using namespace std;

int LRNN::total_layers = 0;

LRNN::LRNN(vector<Layer *> parent, int units, string activation, bool use_bias, bool bidirectional, string name, int dev, int mem) : MLayer(name, dev, mem) {

    this->units = units;
    this->use_bias = use_bias;
    this->bidirectional = bidirectional;
    this->activation=activation;

    isrecurrent=true;

    if (parent[0]->output->ndim != 2) msg("LRNN only works over 2D tensors", "LRNN");

    if(name.empty()) this->name = "RNN" + to_string(++total_layers);

    input = parent[0]->output;
    output = new Tensor(vector<int>{input->shape[0], units}, dev);
    preoutput = new Tensor(vector<int>{input->shape[0], units}, dev);

    // From parent layer
    Wx = new Tensor(vector<int>{input->shape[1], units}, dev);
    params.push_back(Wx);

    gWx = new Tensor(vector<int>{input->shape[1], units}, dev);
    gradients.push_back(gWx);

    // From t-1 RNN
    Wy = new Tensor(vector<int>{units, units}, dev);
    params.push_back(Wy);

    gWy = new Tensor(vector<int>{units, units}, dev);
    gradients.push_back(gWy);


    if (use_bias) {
        bias = new Tensor(vector<int>{units}, dev);
        params.push_back(bias);
        gbias = new Tensor(vector<int>{units}, dev);
        gradients.push_back(gbias);
    }


    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}

LRNN::~LRNN(){
    delete preoutput;
}

// virtual
void LRNN::forward() {
    if (preoutput->size!=output->size)
        preoutput->resize(output->shape[0]);

    Tensor::mult2D(parent[0]->output, 0, Wx, 0, preoutput, 0);
    if (parent.size()>1)
        Tensor::mult2D(parent[1]->output, 0, Wy, 0, preoutput, 1);
    if (use_bias) Tensor::sum2D_rowwise(preoutput, bias, preoutput);

    if (activation == "relu"){
        tensorNN::ReLu(preoutput, output);
    }else if (activation == "sigmoid"){
        tensorNN::Sigmoid(preoutput, output);
    }else if (activation == "hard_sigmoid"){
        tensorNN::HardSigmoid(preoutput, output);
    }else if (activation == "tanh"){
        tensorNN::Tanh(preoutput, output);
    }else if (activation == "none") {
        Tensor::copy(preoutput,output);
    }else {
        msg("Activation not supported for RNN","RNN::RNN");
    }

}

void LRNN::backward() {
    //get gradients with provided delta
    Tensor *daux=new Tensor(delta->shape,delta->device);
    daux->fill_(0.0);

    if (activation == "relu"){
        tensorNN::D_ReLu(delta, preoutput, daux);
        Tensor::copy(daux,delta);
    }else if (activation == "sigmoid"){
        tensorNN::D_Sigmoid(delta, output, daux);
        Tensor::copy(daux,delta);
    }else if (activation == "hard_sigmoid"){
        tensorNN::D_HardSigmoid(delta, preoutput, daux);
        Tensor::copy(daux,delta);
    }else if (activation == "tanh"){
        tensorNN::D_Tanh(delta, output, daux);
        Tensor::copy(daux,delta);
    }

   delete daux;

    if (trainable) {
        Tensor::mult2D(parent[0]->output, 1, delta, 0, gWx, 1);
        if (parent.size()>1)
            Tensor::mult2D(parent[1]->output, 1, delta, 0, gWy, 1);
        if (use_bias) Tensor::reduce_sum2D(delta, gbias, 0, 1);

    }

    Tensor::mult2D(delta, 0, Wx, 1, parent[0]->delta, 1);
    if (parent.size()>1)
        Tensor::mult2D(delta, 0, Wy, 1, parent[1]->delta, 1);


    // Regularizer
    if (trainable) if(reg != nullptr) {reg->apply(this->Wx);reg->apply(this->Wy);}

}


Layer *LRNN::share(int c, int bs, vector<Layer *> p) {
    LRNN *n = new LRNN(p, units, activation, use_bias, bidirectional, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;
    n->do_deletes = false;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->Wx = params[0];
    n->Wy = params[1];
    if (use_bias) n->bias = params[2];

    n->params.push_back(n->Wx);
    n->params.push_back(n->Wy);
    if (use_bias) n->params.push_back(n->bias);

    //share gradients
    for (int i = 0; i < n->gradients.size(); i++) delete n->gradients[i];
    n->gradients.clear();

    n->gWx = gradients[0];
    n->gWy = gradients[1];
    if (use_bias) n->gbias = gradients[2];

    n->gradients.push_back(n->gWx);
    n->gradients.push_back(n->gWy);
    if (use_bias) n->gradients.push_back(n->gbias);

    if (n->reg != nullptr) delete n->reg;
    n->reg = reg;
    if (n->init != nullptr) delete n->init;
    n->init = init;

    return n;
}

Layer *LRNN::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRNN *n = new LRNN(p, units, activation, use_bias, bidirectional,  "clone_" + name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRNN::plot(int c) {
    string s;

    // TODO: Twice the same?
    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Orange,shape=polygon]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Orange,shape=polygon]";

    return s;
}
