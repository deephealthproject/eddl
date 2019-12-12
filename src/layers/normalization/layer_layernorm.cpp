/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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


LLayerNorm::LLayerNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev) : LinLayer(name, dev) {

    input=parent->output;

    Tensor *A;

    if (input->ndim == 2) PD=new PermuteDescriptor({1,0});
    else PD=new PermuteDescriptor({1,0,2,3});

    if (input->ndim == 2) PD2=new PermuteDescriptor({1,0});
    else PD2=new PermuteDescriptor({1,0,2,3});

    PD->build(input->shape);

    A=new Tensor(PD->oshape,dev);

    PD2->build(A->getShape());


    if (input->ndim == 2) {axis.push_back(0);shape.push_back(A->shape[1]);}
    else if (input->ndim == 4) {axis.push_back(0);axis.push_back(2);axis.push_back(3);shape.push_back(A->shape[1]);}
    else {
      input->info();
      msg("LLayerNorm only works over 1D (Dense) or 2D (Conv) tensors","LLayerNorm");
    }

    MD=new MapReduceDescriptor(A,axis);

    delete A;

    if(name.empty()) this->name = "layernorm" + to_string(++total_layers);

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
void LLayerNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    output->resize(batch);
    delta->resize(batch);
    if (target!=nullptr) target->resize(batch);

    delete MD;
    delete PD;
    delete PD2;

    if (input->ndim == 2) PD=new PermuteDescriptor({1,0});
    else PD=new PermuteDescriptor({1,0,2,3});

    if (input->ndim == 2) PD2=new PermuteDescriptor({1,0});
    else PD2=new PermuteDescriptor({1,0,2,3});

    PD->build(input->shape);

    Tensor *A=new Tensor(PD->oshape,dev);

    PD2->build(A->getShape());

    MD=new MapReduceDescriptor(A,axis);

    bn_mean->resize(batch);
    bn_var->resize(batch);
    mean->resize(batch);
    variance->resize(batch);

    delete A;
  }
}


void LLayerNorm::forward() {

  Tensor *A;
  Tensor *B;

  A=new Tensor(PD->oshape,dev);

  Tensor::select(input,A, PD);
  B=new Tensor(A->getShape(),A->device);

  BN_forward(A,B,MD,bn_mean,bn_var,mean,variance,momentum,epsilon,mode==TRMODE);

  Tensor::select(B,output, PD2);

  delete A;
  delete B;
}

void LLayerNorm::backward()
{

  Tensor *A;
  Tensor *B;
  Tensor *C;


  A=new Tensor(PD->oshape,dev);
  Tensor::select(delta,A, PD);



  B=new Tensor(PD->oshape,dev);
  Tensor::select(input,B, PD);

  C=new Tensor(A->getShape(),A->device);
  C->fill_(0.0);

  BN_backward(B,A,C,MD,bn_mean,bn_var,mean,variance,epsilon);

  delete A;

  A=new Tensor(delta->getShape(),dev);
  Tensor::select(C,A, PD2);

  Tensor::inc(A,parent[0]->delta);

  delete A;
  delete B;
  delete C;

}



Layer *LLayerNorm::share(int c, int bs, vector<Layer *> p) {
    LLayerNorm *n = new LLayerNorm(p[0], momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LLayerNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LLayerNorm *n = new LLayerNorm(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev);
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
