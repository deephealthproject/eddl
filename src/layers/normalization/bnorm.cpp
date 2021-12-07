/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/layers/normalization/layer_normalization.h"


using namespace std;


// Different reductions operations using matrices routines
// CuBlas in GPU and Eigen for CPU
void rsum(Tensor *A, Tensor *b, Tensor *ones, Tensor *mem,int p)
{
  int s;

  s=A->shape[p];

  if (p) b->reshape_({1,s});
  else b->reshape_({s,1});

  if (p) Tensor::mult2D(ones,0,b,0,mem,0);
  else Tensor::mult2D(b,0,ones,0,mem,0);

  Tensor::add(1.0,A,1.0,mem,A,0);
  b->reshape_({s});

}

void rdiff(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem,int p)
{
  int s;

  s=A->shape[p];

  if (p) b->reshape_({1,s});
  else b->reshape_({s,1});

  if (p) Tensor::mult2D(ones,0,b,0,mem,0);
  else Tensor::mult2D(b,0,ones,0,mem,0);

  Tensor::add(1.0,A,-1.0,mem,A,0);
  b->reshape_({s});

}

void rmult(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem,int p)
{
  int s;

  s=A->shape[p];


  if (p) b->reshape_({1,s});
  else b->reshape_({s,1});

  if (p) Tensor::mult2D(ones,0,b,0,mem,0);
  else Tensor::mult2D(b,0,ones,0,mem,0);

  Tensor::el_mult(A,mem,A,0);
  b->reshape_({s});

}

void rdiv(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem,int p)
{
  int s;

  s=A->shape[p];


  if (p) b->reshape_({1,s});
  else b->reshape_({s,1});

  if (p) Tensor::mult2D(ones,0,b,0,mem,0);
  else Tensor::mult2D(b,0,ones,0,mem,0);

  Tensor::el_div(A,mem,A,0);
  b->reshape_({s});

}

void cmean(Tensor *A, Tensor *b,Tensor *ones,int p)
{
  int N,M;

  N=A->shape[1-p];
  M=A->shape[p];

  if (p) {
    ones->reshape_({1,N});
    b->reshape_({1,M});
  }
  else {
    ones->reshape_({N,1});
    b->reshape_({M,1});
  }

  if (p) Tensor::mult2D(ones,0,A,0,b,0);
  else Tensor::mult2D(A,0,ones,0,b,0);
  b->div_(N);

  if (p) {
    b->reshape_({M});
    ones->reshape_({N,1});
  }
  else {
    b->reshape_({M});
    ones->reshape_({1,N});
  }
}


void BN_forward(Tensor *input,Tensor *bn_mean, Tensor *bn_var, Tensor *mean, Tensor *variance,float momentum, float epsilon, int trmode)
{
  // General 2D BN
  // NxM tensors where N is thre reduced dimensions to M statistics

  int N,M;

  N=input->shape[0];
  M=input->shape[1];

  Tensor *var=new Tensor({N,M},input->device);
  Tensor *ones=new Tensor({N,1},input->device);
  ones->fill_(1.0);

  if (trmode) {

    Tensor *temporary=new Tensor({N,M},input->device);
    Tensor::copy(input, temporary);

    // mean
    cmean(input, bn_mean, ones);

    // in = in - mean
    rdiff(input, bn_mean, ones, var);

    Tensor::copy(input, var);

    // variance
    var->sqr_();
    cmean(var, bn_var, ones);

    // Update global statistics
    if (momentum != 0.0) {
        Tensor::add(momentum, mean,     (1.0 - momentum), bn_mean, mean,     0);
        Tensor::add(momentum, variance, (1.0 - momentum), bn_var,  variance, 0);
    }

    // sd = sqrt(var + epsilon)
    bn_var->add_(epsilon);
    bn_var->sqrt_();

    // x = in - mean
    rdiff(input, bn_mean, ones, var);
    // x = x / sd
    rdiv(input, bn_var, ones, var);
    // now: in = (in - mean) / sd

  } else {

    Tensor::copy(variance, bn_var);
    bn_var->add_(epsilon);
    bn_var->sqrt_();
    // x = in - global_mean
    rdiff(input, mean, ones, var);
    // x = x / global_sd
    rdiv(input, bn_var, ones, var);
    // now: in = (in - global_mean) / global_sd
  }


  // Free
  delete var;
  delete ones;
}

void BN_backward(Tensor *delta,Tensor *bn_var, Tensor *opa)
{
  // General 2D BN
  // NxM tensors where N is thre reduced dimensions to M statistics

  int N,M;

  N=delta->shape[0];
  M=delta->shape[1];

  Tensor *A=new Tensor({N,M},delta->device);
  Tensor *ones=new Tensor({N},delta->device);
  ones->fill_(1.0);
  Tensor *m=new Tensor({1,M},delta->device);

  // From https://github.com/BVLC/caffe/blob/master/src/caffe/layers/batch_norm_layer.cu
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)./ sqrt(var(X) + eps)
  //          6   4         5   2          1        3      7
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  //1
  Tensor::el_mult(delta,opa,A,0);

  //2
  cmean(A,m,ones);

  //3
  rmult(opa,m,ones,A);

  //4
  cmean(delta,m,ones);

  //5
  rsum(opa,m,ones,A);

  // 6
  Tensor::add(1,delta,-1,opa,delta,0);

  // from forward bn_var=sqrt(var(X) + eps
  // 7
  rdiv(delta,bn_var,ones,A);


  delete ones;
  delete m;
  delete A;

}
