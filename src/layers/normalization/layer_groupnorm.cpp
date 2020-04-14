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

#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/reductions/layer_reductions.h"
#include "eddl/layers/operators/layer_operators.h"

using namespace std;

int LGroupNorm::total_layers = 0;


LGroupNorm::LGroupNorm(Layer *parent, int g, float epsilon, bool affine, string name, int dev, int mem) : LinLayer(name, dev, mem) {

    input=parent->output;
    groups=g;
    this->affine=affine;

    if (input->ndim != 4) {
      input->info();
      msg("LGroupNorm only works over 2D (Conv) tensors","LGroupNorm");
    }

    if (input->shape[1]<groups)
      msg("incorrect group value larger than channels","LGroupNorm");

    if (input->shape[1]%groups)
      msg("incorrect group value not channel divider","LGroupNorm");

    shape.push_back(input->shape[0]*groups);

    this->epsilon = epsilon;

    output=new Tensor(input->getShape(),dev);
    opa=new Tensor(input->getShape(),dev);
    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);

    if (affine) {
      int s=input->shape[1]/groups;

      bn_g=new Tensor({s},dev);
      bn_b=new Tensor({s},dev);

      gbn_g=new Tensor({s},dev);
      gbn_b=new Tensor({s},dev);

      params.push_back(bn_g);
      params.push_back(bn_b);

      gradients.push_back(gbn_g);
      gradients.push_back(gbn_b);
    }


    parent->addchild(this);
    addparent(parent);
}

// virtual
void LGroupNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    opa->reshape_(output->getShape());

    output->resize(batch);
    opa->resize(batch);

    bn_mean->resize(batch*groups);
    bn_var->resize(batch*groups);

  }
}

// override functions:
int LGroupNorm::get_trainable_params_count()
{
  if (affine) return 2;  // only 2 trainable params
  else return 0;  // no trainable params
}

void LGroupNorm::initialize() {
  if (affine) {
    params[0]->fill_(1.0);
    params[1]->fill_(0.0);
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

  // Now is like a LayerNorm but affine is channel-wise
  M=b=input->shape[0];
  z=input->shape[1];
  r=input->shape[2];
  c=input->shape[3];
  N=z*r*c;

  Tensor *in=new Tensor(input->getShape(),input->device);

  permute_batch_last(input,in);
  in->reshape_({N,M}); // now is a 2D tensor
  opa->reshape_({N,M});

  BN_forward(in,bn_mean,bn_var,nullptr,nullptr,0.0,epsilon,1);
  Tensor::copy(in,opa);

  if (affine) {
    Tensor *in2=new Tensor({b,z,r,c},input->device);
    int M2,N2;
    N2=b*r*c;
    M2=z;

    permute_batch_first(opa,in2);
    permute_channels_last(in2,in);

    in->reshape_({N2,M2});

    Tensor *var=new Tensor({N2,M2},input->device);
    Tensor *ones=new Tensor({N2,1},input->device);
    ones->fill_(1.0);

    // apply affine transform in=gamma*in+beta
    rmult(in,bn_g,ones,var);
    rsum(in,bn_b,ones,var);

    in2->reshape_({b,z,r,c});
    permute_channels_first(in,in2);
    permute_batch_last(in2,in);

    in->reshape_({N,M});

    delete in2;
    delete var;
    delete ones;
  }

  // copy in to ouput
  output->reshape_({b,z,r,c});
  permute_batch_first(in,output);

  input->reshape_({b/groups,z*groups,r,c});
  output->reshape_({b/groups,z*groups,r,c});

  delete in;
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

  // Now is like a LayerNorm but affine is channel-wise
  M=b=delta->shape[0];
  z=delta->shape[1];
  r=delta->shape[2];
  c=delta->shape[3];

  N=z*r*c;

  // permute input and delta
  dp=new Tensor({N,M},input->device);
  permute_batch_last(delta,dp);


  // Affine
  if (affine) {
    int M2,N2;
    N2=b*r*c;
    M2=z;

    Tensor *A=new Tensor({N,M},delta->device);
    Tensor *A2=new Tensor({N,M},delta->device);
    Tensor *ones=new Tensor({N2},delta->device);
    ones->fill_(1.0);
    Tensor *m=new Tensor({1,M2},delta->device);
    //1 gamma
    Tensor::el_mult(dp,opa,A,0);

    A2->reshape_({b,z,r,c});
    permute_batch_first(A,A2);
    permute_channels_last(A2,A);
    A->reshape_({N2,M2});

    cmean(A,m,ones);
    Tensor::add(1,gbn_g,1,m,gbn_g,0);

    //2 Beta
    A2->reshape_({b,z,r,c});
    permute_batch_first(dp,A2);
    permute_channels_last(A2,A);
    A->reshape_({N2,M2});

    cmean(A,m,ones);
    Tensor::add(1,gbn_b,1,m,gbn_b,0);

    // delta=dE/dY
    // Obtain dE/dY from delta:
    A2->reshape_({N2,M2});
    rmult(A,bn_g,ones,A2);

    A2->reshape_({b,z,r,c});
    dp->reshape_({b,z,r,c});
    permute_channels_first(A,A2);
    permute_batch_last(A2,dp);
    dp->reshape_({N,M});

    delete A;
    delete A2;
    delete ones;
    delete m;
  }

  BN_backward(dp,bn_var,opa);

  permute_batch_first(dp,delta);

  delta->reshape_({b/groups,z*groups,r,c});
  Tensor::inc(delta, parent[0]->delta);



  delete dp;
}



Layer *LGroupNorm::share(int c, int bs, vector<Layer *> p) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, epsilon, affine,"share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LGroupNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, epsilon, affine,"clone_" + to_string(todev) + name, todev, this->mem_level);
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
