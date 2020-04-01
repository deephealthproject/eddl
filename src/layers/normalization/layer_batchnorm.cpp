/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "layers/normalization/layer_normalization.h"
#include "layers/reductions/layer_reductions.h"
#include "layers/operators/layer_operators.h"

using namespace std;

int LBatchNorm::total_layers = 0;


LBatchNorm::LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev, int mem) : LinLayer(name, dev, mem) {
    input=parent->output;

    if (input->ndim == 2) {axis.push_back(0);shape.push_back(input->shape[1]);}
    else if (input->ndim == 4) {axis.push_back(0);axis.push_back(2);axis.push_back(3);shape.push_back(input->shape[1]);}
    else {
        input->info();
        msg("LBatchNorm only works over 1D (Dense) or 2D (Conv) tensors","LBatchNorm");
    }


    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
//    if (!mem_level) delta=new Tensor(input->getShape(),dev);

    mean=new Tensor(shape,dev);
    mean->fill_(0.0);
    variance=new Tensor(shape,dev);
    variance->fill_(1.0);

    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);

    if (affine) {
      bn_g=new Tensor(shape,dev);
      bn_b=new Tensor(shape,dev);

      gbn_g=new Tensor(shape,dev);
      gbn_b=new Tensor(shape,dev);

      opa=new Tensor(output->getShape(),dev); //output pre-affine

      params.push_back(bn_g);
      params.push_back(bn_b);

      gradients.push_back(gbn_g);
      gradients.push_back(gbn_b);
    }

    // no trainable:
    params.push_back(mean);
    params.push_back(variance);

    MD=new MapReduceDescriptor(input,axis);


    parent->addchild(this);
    addparent(parent);
}


// override functions:
int LBatchNorm::get_trainable_params_count()
{
  if (affine) return 2;  // only 2 trainable params
  else return 0;  // no trainable params
}

void LBatchNorm::initialize() {
  if (affine) {
    params[0]->fill_(1.0);
    params[1]->fill_(0.0);
  }
}

void LBatchNorm::resize(int batch){
    if (batch!=output->shape[0]) {
        output->resize(batch);
        if (affine) opa->resize(batch);
//        if (!mem_level) delta->resize(batch);
        
        delete MD;
        MD=new MapReduceDescriptor(input,axis);
    }
}




void LBatchNorm::forward() {
    BN_forward(input,output,MD,bn_mean,bn_var,mean,variance,momentum,epsilon,affine,bn_g,bn_b,opa,mode==TRMODE);
}

void LBatchNorm::backward(){
    BN_backward(input,delta, parent[0]->delta,MD,bn_mean,bn_var,mean,variance,epsilon,affine,bn_g,bn_b,gbn_g,gbn_b,opa);
}



Layer *LBatchNorm::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LBatchNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev,mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LBatchNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
