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


LGroupNorm::LGroupNorm(Layer *parent, int g, float momentum, float epsilon, bool affine, string name, int dev) : LinLayer(name, dev) {

    input=parent->output;
    groups=g;

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

    axis.push_back(0);axis.push_back(2);axis.push_back(3);shape.push_back(B->shape[1]);

    MD=new MapReduceDescriptor(B,axis);

    delete A;
    delete B;

    if(name.empty()) this->name = "groupnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
    delta=new Tensor(input->getShape(),dev);

    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);

    if (momentum!=0.0) {
        mean=new Tensor(shape,dev);
        mean->fill_(0.0);

        variance=new Tensor(shape,dev);
        variance->fill_(1.0);
    }


    parent->addchild(this);
    addparent(parent);
}


// virtual
void LGroupNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    output->resize(batch);
    delta->resize(batch);
    if (target!=nullptr) target->resize(batch);
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

    delete A;
    delete B;


    bn_mean->resize(N*groups);
    bn_var->resize(N*groups);
    mean->resize(N*groups);
    variance->resize(N*groups);


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

  BN_forward(B,C,MD,bn_mean,bn_var,mean,variance,momentum,epsilon,mode==TRMODE);

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

  BN_backward(C,B,A,MD,bn_mean,bn_var,mean,variance,epsilon);

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
    LGroupNorm *n = new LGroupNorm(p[0], groups, momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LGroupNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LGroupNorm *n = new LGroupNorm(p[0], groups, momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev);
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
