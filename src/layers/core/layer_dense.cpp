/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer_core.h"

using namespace std;

int LDense::total_layers = 0;

LDense::LDense(Layer *parent, int ndim, bool use_bias, string name, int dev) : LinLayer(name, dev) {
    if (parent->output->ndim != 2) msg("LDense only works over 2D tensors", "LDense");

    if(name.empty()) this->name = "dense" + to_string(++total_layers);
    this->ndim = ndim;
    this->use_bias = use_bias;


    input = parent->output;
    output = new Tensor(vector<int>{input->shape[0], ndim}, dev);
    delta = new Tensor(output->getShape(), dev);

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


// virtual
void  LDense::resize(int batch){
  Layer::resize(batch);
}

void LDense::forward() {
    Tensor::mult2D(input, 0, W, 0, output, 0);
    if (use_bias) Tensor::sum2D_rowwise(output, bias, output);
}

void LDense::backward() {

    //get gradients with provided delta
    if (trainable) {
      Tensor::mult2D(input, 1, delta, 0, gW, 1);
      if (use_bias) Tensor::reduce_sum2D(delta, gbias, 0, 1);
    }

    // backprop delta
    if (parent.size()) {
        //1: note that increment parent delta
        Tensor::mult2D(delta, 0, W, 1, parent[0]->delta, 1);
    }

    // Regularizer
    if (trainable) if(reg != nullptr) {reg->apply(this->W);}
}

void LDense::update_weights(Tensor* w, Tensor* bias) {
	Tensor::copy( w, this->W );
	if ( bias != nullptr ) Tensor::copy( bias, this->bias );
}

void LDense::accumulate_accumulated_gradients(Tensor* gw, Tensor* gbias) {
	W->add_( gw );
	if ( gbias != nullptr ) bias->add_( gbias );
	
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
    LDense *n = new LDense(p[0], ndim, use_bias, "share_" + to_string(c) + name, dev);
    n->orig = this;

    //share params
    for (int i = 0; i < n->params.size(); i++) delete n->params[i];
    n->params.clear();

    n->W = params[0];
    if (use_bias) n->bias = params[1];

	if ( distributed_training ) {
		n->acc_gW = this->acc_gradients[0];
		if ( use_bias ) n->acc_gbias = this->acc_gradients[1];
	}

    n->params.push_back(n->W);
    if (use_bias) n->params.push_back(n->bias);

    n->reg=reg;
    n->init=init;

    return n;
}

Layer *LDense::clone(int c, int bs, vector<Layer *> p, int todev) {
    LDense *n = new LDense(p[0], ndim, use_bias, "clone_" + to_string(todev) + name, todev);
    n->orig = this;

    n->reg=reg;
    n->init=init;

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

	if ( distributed_training ) {
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
}
