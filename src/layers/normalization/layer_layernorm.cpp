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

int LLayerNorm::total_layers = 0;


LLayerNorm::LLayerNorm(Layer *parent,  float epsilon, bool affine,  string name, int dev, int mem) : LinLayer(name, dev, mem) {
    input=parent->output;
    this->affine=affine;

    shape.push_back(input->shape[0]);

    if ((input->ndim != 2)&&(input->ndim != 4)) {
        input->info();
        msg("LBatchNorm only works over 1D (Dense) or 2D (Conv) tensors","LBatchNorm");
    }


    if(name.empty()) this->name = "layernorm" + to_string(++total_layers);

    this->epsilon = epsilon;

    output=new Tensor(input->getShape(),dev);
    opa=new Tensor(input->getShape(),dev);
    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);


    if (affine) {
      int size=input->size/input->shape[0];//z*r*c

      //https://pytorch.org/docs/stable/nn.html#torch.nn.LayerNorm
      //Unlike Batch Normalization which applies scalar scale and bias
      //for each entire channel/plane with the affine option,
      //Layer Normalization applies per-element scale and bias with
      //elementwise_affine.
      bn_g=new Tensor({size},dev);
      bn_b=new Tensor({size},dev);
      gbn_g=new Tensor({size},dev);
      gbn_b=new Tensor({size},dev);

      params.push_back(bn_g);
      params.push_back(bn_b);

      gradients.push_back(gbn_g);
      gradients.push_back(gbn_b);
    }


    parent->addchild(this);
    addparent(parent);
}

void LLayerNorm::resize(int batch){
    if (batch!=output->shape[0]) {
        opa->reshape_(output->getShape());
        output->resize(batch);
        opa->resize(batch);

        bn_mean->resize(batch);
        bn_var->resize(batch);
    }
}

// override functions:
int LLayerNorm::get_trainable_params_count()
{
  if (affine) return 2;  // only 2 trainable params
  else return 0;  // no trainable params
}

void LLayerNorm::initialize() {
  if (affine) {
    params[0]->fill_(1.0);
    params[1]->fill_(0.0);
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

  Tensor *in;
  if (input->ndim==2) {
    M=b=input->shape[0];
    N=d=input->shape[1];

    in=new Tensor({b*d},input->device);
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

    in=new Tensor({b*r*c*z},input->device);
    permute_batch_last(input,in);
    in->reshape_({N,M}); // now is a 2D tensor
    opa->reshape_({N,M});

  }

  BN_forward(in,bn_mean,bn_var,nullptr,nullptr,0.0,epsilon,1);
  Tensor::copy(in,opa);

  if (affine) {
    Tensor *var=new Tensor({N,M},input->device);
    Tensor *ones=new Tensor({1,M},input->device);
    ones->fill_(1.0);

    // apply affine transform in=gamma*in+beta
    rmult(in,bn_g,ones,var,0);
    rsum(in,bn_b,ones,var,0);
    delete var;
    delete ones;
  }

  // copy in to ouput
  if (input->ndim==4) {permute_batch_first(in,output);}
  else {
    output->reshape_({b,d,1,1});
    permute_batch_first(in,output);
    output->reshape_({b,d});
  }

  delete in;


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
  // Affine

  if (affine) {
    Tensor *A=new Tensor({N,M},delta->device);
    Tensor *ones=new Tensor({1,M},delta->device);
    ones->fill_(1.0);
    Tensor *m=new Tensor({1,N},delta->device);

    //1 gamma
    Tensor::el_mult(dp,opa,A,0);
    cmean(A,m,ones,0);
    Tensor::add(1,gbn_g,1,m,gbn_g,0);

    //2 Beta
    cmean(dp,m,ones,0);
    Tensor::add(1,gbn_b,1,m,gbn_b,0);

    // delta=dE/dY
    // Obtain dE/dY from delta:

    rmult(dp,bn_g,ones,A,0);

    delete A;
    delete ones;
    delete m;
  }

  BN_backward(dp,bn_var,opa);

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
    LLayerNorm *n= new LLayerNorm(p[0], epsilon, affine, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LLayerNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LLayerNorm *n= new LLayerNorm(p[0], epsilon, affine, "clone_" + to_string(todev) + name, todev, this->mem_level);
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
