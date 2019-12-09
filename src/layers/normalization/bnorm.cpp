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


using namespace std;

void BN_forward(Tensor *input,Tensor *output, MapReduceDescriptor *MD, Tensor *bn_mean, Tensor *bn_var, Tensor *mean, Tensor *variance,float momentum, float epsilon, int trmode)
{

  if (trmode) {

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

    Tensor *sd=bn_var->clone();

    sd->add_(epsilon);
    sd->sqrt_();

    reduce_div(output,sd,MD);
    delete osqr;
    delete sd;

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

void BN_backward(Tensor* input, Tensor *delta,Tensor *pdelta, MapReduceDescriptor *MD, Tensor *bn_mean, Tensor *bn_var, Tensor *mean, Tensor *variance,float epsilon)
{
  // from http://proceedings.mlr.press/v37/ioffe15.pdf

  int m;

  if (delta->ndim == 2)
    m=delta->shape[0];
  else
    m=delta->shape[0]*delta->shape[2]*delta->shape[3];

  Tensor *dmean=new Tensor(bn_mean->getShape(),bn_mean->device);
  Tensor *dvar=new Tensor(bn_var->getShape(),bn_var->device);

  Tensor *X=input->clone();
  Tensor *Tvar32=bn_var->clone();
  Tensor *Tsqvar=bn_var->clone();
  Tensor *dx_hat=delta->clone();

  // No affine


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

  Tensor::copy(delta,pdelta);


  delete X;
  delete Tvar32;
  delete Tsqvar;
  delete dx_hat;
  delete dvar;
  delete dmean;

}
