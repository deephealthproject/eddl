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

#include "eddl/layers/recurrent/layer_recurrent.h"


using namespace std;

int LRNN::total_layers = 0;

LRNN::LRNN(vector<Layer *> parent, int units, int num_layers, bool use_bias, float dropout, bool bidirectional, string name, int dev, int mem) : MLayer(name, dev, mem) {

    this->units = units;
    int ndim=units;
    this->num_layers = num_layers;
    this->use_bias = use_bias;
    this->dropout = dropout;
    this->bidirectional = bidirectional;
    isrecurrent=true;

    // TODO: Implement

    if (parent[0]->output->ndim != 2) msg("LRNN only works over 2D tensors", "LRNN");

    if(name.empty()) this->name = "RNN" + to_string(++total_layers);

    input = parent[0]->output;
    output = new Tensor(vector<int>{input->shape[0], ndim}, dev);

    // From parent layer
    Wx = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    params.push_back(Wx);

    gWx = new Tensor(vector<int>{input->shape[1], ndim}, dev);
    gradients.push_back(gWx);

    // From t-1 RNN
    Wy = new Tensor(vector<int>{ndim, ndim}, dev);
    params.push_back(Wy);

    gWy = new Tensor(vector<int>{ndim, ndim}, dev);
    gradients.push_back(gWy);


    if (use_bias) {
      bias = new Tensor(vector<int>{ndim}, dev);
      params.push_back(bias);
      gbias = new Tensor(vector<int>{ndim}, dev);
      if (use_bias) gradients.push_back(gbias);
    }


    for (int i = 0; i < parent.size(); ++i) {
        parent[i]->addchild(this);
        addparent(parent[i]);
    }

}


// virtual
void LRNN::forward() {
  Tensor::mult2D(parent[0]->output, 0, Wx, 0, output, 0);
  if (parent.size()>1)
    Tensor::mult2D(parent[1]->output, 0, Wy, 0, output, 1);
  if (use_bias) Tensor::sum2D_rowwise(output, bias, output);
}

void LRNN::backward() {
  //get gradients with provided delta

  if (trainable) {
      Tensor::mult2D(parent[0]->output, 1, delta, 0, gWx, 1);
      if (parent.size()>1)
        Tensor::mult2D(parent[1]->output, 1, delta, 0, gWy, 1);
    if (use_bias) Tensor::reduce_sum2D(delta, gbias, 0, 1);
  }

  //1: note that increment parent delta
  Tensor::mult2D(delta, 0, Wx, 1, parent[0]->delta, 1);
  if (parent.size()>1)
    Tensor::mult2D(delta, 0, Wy, 1, parent[1]->delta, 1);

  // Regularizer
  if (trainable) if(reg != nullptr) {reg->apply(this->Wx);reg->apply(this->Wy);}

}


Layer *LRNN::share(int c, int bs, vector<Layer *> p) {
    LRNN *n = new LRNN(p, units, num_layers, use_bias, dropout, bidirectional, "share_"+to_string(c)+this->name, this->dev, this->mem_level);
    n->orig = this;
    n->isshared=true;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->Wx = params[0];
    n->Wy = params[1];
    if (use_bias) n->bias = params[2];

    n->params.push_back(n->Wx);
    n->params.push_back(n->Wy);
    if (use_bias) n->params.push_back(n->bias);

    n->reg=reg;
    n->init=init;

    return n;
}

Layer *LRNN::clone(int c, int bs, vector<Layer *> p, int todev) {
    LRNN *n = new LRNN(p, units, num_layers, use_bias, dropout, bidirectional,  "clone_" + name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LRNN::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Orange,shape=polygon]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=Orange,shape=polygon]";

    return s;
}
