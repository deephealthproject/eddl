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

int LGroupNorm::total_layers = 0;


LGroupNorm::LGroupNorm(Layer *parent, int g, float epsilon, string name, int dev, int mem) : LinLayer(name, dev, mem) {

    input=parent->output;
    groups=g;

    if (input->ndim != 4) {
      input->info();
      msg("LGroupNorm only works over 2D (Conv) tensors","LGroupNorm");
    }

    if (input->shape[1]%groups) msg("incorrect group value not channel divider","LGroupNorm");

    shape.push_back(input->shape[0]*groups);

    this->epsilon = epsilon;

    output=new Tensor(input->getShape(),dev);
    in=new Tensor(input->getShape(),dev);
    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);

    parent->addchild(this);
    addparent(parent);
}

// virtual
void LGroupNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    in->reshape_(output->getShape());

    output->resize(batch);
    in->resize(batch);

    bn_mean->resize(batch*groups);
    bn_var->resize(batch*groups);

  }
}

void LGroupNorm::forward()
{
  // Input = Output = {Batch,Channels,H,W} OR {Batch,Dim}
  // bn_mean = bn_var = mean = variance = bn_g = bn_b = {Batch}

  int M,N;
  int b,z,r,c,d;

  b=input->shape[0];
  z=input->shape[1];
  r=input->shape[2];
  c=input->shape[3];

  input->reshape_({b*groups,z/groups,r,c});

  // Now is like a LayerNorm
  M=b=input->shape[0];
  z=input->shape[1];
  r=input->shape[2];
  c=input->shape[3];
  N=z*r*c;

  permute_batch_last(input,in);
  in->reshape_({N,M}); // now is a 2D tensor
  input->reshape_({b/groups,z*groups,r,c});

  BN_forward(in,bn_mean,bn_var,nullptr,nullptr,0.0,epsilon,false,nullptr,nullptr,nullptr,1);

  // copy in to ouput
  permute_batch_first(in,output);
}

void LGroupNorm::backward()
{
  int M,N;
  int b,z,r,c,d;

  Tensor *dp;

  b=delta->shape[0];
  z=delta->shape[1];
  r=delta->shape[2];
  c=delta->shape[3];

  delta->reshape_({b*groups,z/groups,r,c});

  // Now is like a LayerNorm
  M=b=delta->shape[0];
  z=delta->shape[1];
  r=delta->shape[2];
  c=delta->shape[3];

  N=z*r*c;
  
  // permute input and delta
  dp=new Tensor({z,r,c,b},input->device);
  permute_batch_last(delta,dp);
  dp->reshape_({N,M});

  delta->reshape_({b/groups,z*groups,r,c});

  BN_backward(dp,bn_mean,bn_var,nullptr,nullptr,epsilon,false,nullptr,nullptr,nullptr,nullptr,in);


  permute_batch_first(dp,delta);
  Tensor::inc(delta, parent[0]->delta);

  delete dp;
}



Layer *LGroupNorm::share(int c, int bs, vector<Layer *> p) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, epsilon, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LGroupNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, epsilon, "clone_" + to_string(todev) + name, todev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}


string LGroupNorm::plot(int c) {
    string s;

    if (c) s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=bisque4,shape=box]";
    else s = name + " [label=" + "\"" + name + "\",style=filled,fontsize=12,fillcolor=White,shape=box]";

    return s;
}
