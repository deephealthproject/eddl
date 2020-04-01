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

int LGroupNorm::total_layers = 0;


LGroupNorm::LGroupNorm(Layer *parent, int g, float momentum, float epsilon, bool affine, string name, int dev, int mem) : LinLayer(name, dev, mem) {

    input=parent->output;
    groups=g;


    affine=false;

    if (input->ndim != 4) {
      input->info();
      msg("LGroupNorm only works over 2D (Conv) tensors","LGroupNorm");
    }

    N=input->shape[0];
    CH=input->shape[1];
    H=input->shape[2];
    W=input->shape[3];

    if (CH%groups) msg("incorrect group value not channel divider","LGroupNorm");

    Tensor *A=input->clone();
    A->reshape_({N*groups,CH/groups,H,W});

    PD=new PermuteDescriptor({1,0,2,3});
    PD->build(A->shape);

    PD2=new PermuteDescriptor({1,0,2,3});
    Tensor *B=new Tensor(PD->oshape,dev);
    PD2->build(B->getShape());

    axis.push_back(0);axis.push_back(2);axis.push_back(3);
    shape.push_back(B->shape[1]);

    MD=new MapReduceDescriptor(B,axis);


    if (affine) opa=new Tensor(B->getShape(),dev); //output pre-affine


    delete A;
    delete B;

    if(name.empty()) this->name = "groupnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
//    delta=new Tensor(input->getShape(),dev);

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

      params.push_back(bn_g);
      params.push_back(bn_b);

      gradients.push_back(gbn_g);
      gradients.push_back(gbn_b);
    }

    // no trainable:
    params.push_back(mean);
    params.push_back(variance);

    parent->addchild(this);
    addparent(parent);
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

// virtual
void LGroupNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    output->resize(batch);
//    delta->resize(batch);

    delete MD;

    N=batch;

    delete PD;
    delete PD2;

    Tensor *A=input->clone();
    A->reshape_({N*groups,CH/groups,H,W});

    PD=new PermuteDescriptor({1,0,2,3});
    PD->build(A->shape);

    PD2=new PermuteDescriptor({1,0,2,3});
    Tensor *B=new Tensor(PD->oshape,dev);
    PD2->build(B->getShape());

    MD=new MapReduceDescriptor(B,axis);

    if (affine) {
      delete opa;
      opa=new Tensor(B->getShape(),dev); //output pre-affine
    }

    delete A;
    delete B;


    bn_mean->resize(N*groups);
    bn_var->resize(N*groups);
    mean->resize(N*groups);
    variance->resize(N*groups);
    if (affine) {
      bn_g->resize(N*groups);
      bn_b->resize(N*groups);

      gbn_g->resize(N*groups);
      gbn_b->resize(N*groups);

    }

  }
}

void LGroupNorm::forward() {

  Tensor *A;
  Tensor *B;
  Tensor *C;

  A=input->clone();
  A->reshape_({N*groups,CH/groups,H,W});

  B=new Tensor(PD->oshape,dev);
  C=new Tensor(PD->oshape,dev);

  Tensor::select(A,B, PD);

  BN_forward(B,C,MD,bn_mean,bn_var,mean,variance,momentum,epsilon,affine,bn_g,bn_b,opa,mode==TRMODE);

  Tensor::select(C,A, PD2);

  A->reshape_({N,CH,H,W});

  Tensor::copy(A,output);

  delete A;
  delete B;
  delete C;

}

void LGroupNorm::backward()
{
  Tensor *A;
  Tensor *B;
  Tensor *C;


  A=delta->clone();
  A->reshape_({N*groups,CH/groups,H,W});
  B=new Tensor(PD->oshape,dev);
  Tensor::select(A,B, PD);

  delete A;

  A=input->clone();
  A->reshape_({N*groups,CH/groups,H,W});
  C=new Tensor(PD->oshape,dev);
  Tensor::select(A,C, PD);

  delete A;

  A=new Tensor(PD->oshape,dev);
  A->fill_(0.0);


  BN_backward(C,B,A,MD,bn_mean,bn_var,mean,variance,epsilon,affine,bn_g,bn_b,gbn_g,gbn_b,opa);

  delete B;

  B=new Tensor(PD2->oshape,dev);

  Tensor::select(A,B, PD2);
  B->reshape_({N,CH,H,W});

  Tensor::inc(B,parent[0]->delta);

  delete A;
  delete B;
  delete C;
}



Layer *LGroupNorm::share(int c, int bs, vector<Layer *> p) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, momentum, epsilon, affine, "share_" + to_string(c) + this->name, this->dev, this->mem_level);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LGroupNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev, this->mem_level);
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
