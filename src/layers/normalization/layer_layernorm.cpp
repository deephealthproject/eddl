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

#include "layer_normalization.h"
#include "../reductions/layer_reductions.h"
#include "../operators/layer_operators.h"

using namespace std;

int LLayerNorm::total_layers = 0;


LLayerNorm::LLayerNorm(Layer *parent,  float epsilon,  string name, int dev, int mem) : LinLayer(name, dev, mem) {
    input=parent->output;

    shape.push_back(input->shape[0]);

    if ((input->ndim != 2)&&(input->ndim != 4)) {
        input->info();
        msg("LBatchNorm only works over 1D (Dense) or 2D (Conv) tensors","LBatchNorm");
    }


    if(name.empty()) this->name = "layernorm" + to_string(++total_layers);

    this->epsilon = epsilon;

    output=new Tensor(input->getShape(),dev);
    in=new Tensor(input->getShape(),dev);
    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);

    parent->addchild(this);
    addparent(parent);
}

void LLayerNorm::resize(int batch){
    if (batch!=output->shape[0]) {
        in->reshape_(output->getShape());

        output->resize(batch);
        in->resize(batch);
        bn_mean->resize(batch);
        bn_var->resize(batch);
    }
}


// Permute 4D tensors and set N,M values.
// Essentialy 4D Tensors are reshaped as 2D and
// all the batchnorm works over 2D Tensors
void LLayerNorm::forward() {
  // Input = Output = {Batch,Channels,H,W} OR {Batch,Dim}
  // bn_mean = bn_var = mean = variance = bn_g = bn_b = {Batch}

  int M,N;
  int b,z,r,c,d;

  if (input->ndim==2) {
    M=b=input->shape[0];
    N=d=input->shape[1];

    input->reshape_({b,d,1,1});
    permute_batch_last(input,in);
    input->reshape_({M,N});
    in->reshape_({N,M});

  }
  else {
    M=b=input->shape[0];
    z=input->shape[1];
    r=input->shape[2];
    c=input->shape[3];
    N=z*r*c;

    permute_batch_last(input,in);
    in->reshape_({N,M}); // now is a 2D tensor

  }


  BN_forward(in,bn_mean,bn_var,nullptr,nullptr,0.0,epsilon,false,nullptr,nullptr,nullptr,1);

  // copy in to ouput
  if (input->ndim==4) {permute_batch_first(in,output);}
  else {
    output->reshape_({b,d,1,1});
    permute_batch_first(in,output);
    output->reshape_({b,d});
  }



}

void LLayerNorm::backward()
{
  int M,N;
  int b,z,r,c,d;

  Tensor *dp;

  if (input->ndim==2) {
    M=b=delta->shape[0];
    N=d=delta->shape[1];
    delta->reshape_({b,d,1,1});
    dp=new Tensor({d,1,1,b},input->device);
    permute_batch_last(delta,dp);
    delta->reshape_({M,N});
    dp->reshape_({N,M});
  }
  else {
    M=b=input->shape[0];
    z=input->shape[1];
    r=input->shape[2];
    c=input->shape[3];

    N=z*r*c;

    // permute input and delta
    dp=new Tensor({z,r,c,b},input->device);
    permute_batch_last(delta,dp);
    dp->reshape_({N,M});

  }

  BN_backward(dp,bn_mean,bn_var,nullptr,nullptr,epsilon,false,nullptr,nullptr,nullptr,nullptr,in);

  // Inc parent delta
  if (input->ndim==4) {
    permute_batch_first(dp,delta);
    Tensor::inc(delta, parent[0]->delta);
  }
  else {
    delta->reshape_({b,d,1,1});
    permute_batch_first(dp,delta);
    delta->reshape_({b,d});
    Tensor::inc(delta, parent[0]->delta);
  }

  delete dp;

}



Layer *LLayerNorm::share(int c, int bs, vector<Layer *> p) {
    LLayerNorm *n= new LLayerNorm(p[0], epsilon, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LLayerNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LLayerNorm *n= new LLayerNorm(p[0], epsilon, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LLayerNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
