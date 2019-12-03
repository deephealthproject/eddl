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

int LBatchNorm::total_layers = 0;


LBatchNorm::LBatchNorm(Layer *parent, float momentum, float epsilon, bool affine, string name, int dev) : LinLayer(name, dev) {

    input=parent->output;

    if (input->ndim == 2) {axis.push_back(0);shape.push_back(input->shape[1]);}
    else if (input->ndim == 4) {axis.push_back(0);axis.push_back(2);axis.push_back(3);shape.push_back(input->shape[1]);}
    else msg("LBatchNorm only works over 1D (Debse) or 2D (Conv) tensors","LBatchNorm");


    if(name.empty()) this->name = "batchnorm" + to_string(++total_layers);

    this->momentum = momentum;
    this->epsilon = epsilon;
    this->affine = affine;

    output=new Tensor(input->getShape(),dev);
    delta=new Tensor(input->getShape(),dev);

    bn_mean=new Tensor(shape,dev);
    bn_var=new Tensor(shape,dev);
    sd=new Tensor(shape,dev);


    MD=new MapReduceDescriptor(input,axis);

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
void LBatchNorm::resize(int batch){
  if (batch!=output->shape[0]) {
    output->resize(batch);
    delta->resize(batch);
    if (target!=nullptr) target->resize(batch);
    delete MD;
    MD=new MapReduceDescriptor(input,axis);
  }


}

void LBatchNorm::forward() {


  if (mode == TRMODE) {

    Tensor::copy(input,output);

    reduce_mean(output,bn_mean,MD);

    reduce_diff(output,bn_mean,MD);

    Tensor *osqr=output->clone();
    osqr->sqr_();

    reduce_mean(osqr,bn_var,MD);

    if (momentum!=0.0) {
      Tensor::add(momentum, mean, (1.0-momentum), bn_mean,mean,0);
      Tensor::add(momentum, variance, (1.0-momentum), bn_var,variance,0);
    }

    Tensor::copy(bn_var,sd);

    sd->add_(epsilon);
    sd->sqrt_();

    reduce_div(output,sd,MD);
    delete osqr;

  }
  else { // testmode

    reduce_diff(input,mean,MD);
    Tensor::copy(variance,bn_var);
    bn_var->add_(epsilon);
    bn_var->sqrt_();
    reduce_div(input,bn_var,MD);
    Tensor::copy(input,output);
  }

}

void LBatchNorm::backward()
{
  // from http://proceedings.mlr.press/v37/ioffe15.pdf
  // still not affine transform

  int m;
  
  if (input->ndim == 2)
    m=delta->shape[0];
  else
    m=delta->shape[0]*delta->shape[2]*delta->shape[3];

  Tensor *dmean=new Tensor(bn_mean->getShape(),dev);
  Tensor *dvar=new Tensor(bn_var->getShape(),dev);

  Tensor *X=input->clone();
  Tensor *Tvar32=bn_var->clone();
  Tensor *Tsqvar=bn_var->clone();
  Tensor *dx_hat=delta->clone();

  // No affine
  //reduced_mult(Delta,bn_g,dx_hat,0,1);
  //4 Var : dvar
  Tsqvar->add_(epsilon);
  Tsqvar->sqrt_();

  Tvar32->add_(epsilon);
  Tvar32->pow_(1.5);

  X->div_(-1.0);
  reduce_sum(X,bn_mean,MD);
  X->div_(2.0);
  reduce_div(X,Tvar32,MD);
  Tensor::el_mult(X,dx_hat,X,0);
  reduce_mean(X,dvar,MD);
  dvar->mult_(m);


  //5 Mean
  reduce_div(dx_hat,Tsqvar,MD);
  //dx_hat->mult_(-1);
  reduce_mean(dx_hat,dmean,MD);
  dmean->mult_(-m);
  //Tensor::copy(delta, dx_hat);

  //6 Delta
  //reduce_div(dx_hat,Tsqvar,MD);
  Tensor::copy(dx_hat,delta);

  Tensor::copy(input,X);
  X->mult_(2.0/m);
  bn_mean->mult_(-2.0/m);
  reduce_sum(X,bn_mean,MD);
  reduce_mult(X,dvar,MD);
  Tensor::inc(X,delta);
  dmean->mult_(1.0/m);
  reduce_sum(delta,dmean,MD);

  Tensor::copy(delta,parent[0]->delta);


delete X;
delete Tvar32;
delete Tsqvar;
delete dx_hat;
delete dvar;
delete dmean;


}



Layer *LBatchNorm::share(int c, int bs, vector<Layer *> p) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "share_" + to_string(c) + name, dev);
    n->orig = this;

    // TODO: Implement

    return n;
}

Layer *LBatchNorm::clone(int c, int bs, vector<Layer *> p, int todev) {
    LBatchNorm *n = new LBatchNorm(p[0], momentum, epsilon, affine, "clone_" + to_string(todev) + name, todev);
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
