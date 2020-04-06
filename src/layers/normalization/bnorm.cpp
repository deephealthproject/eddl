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


using namespace std;

void rsum(Tensor *A, Tensor *b, Tensor *ones, Tensor *mem)
{
  int N,M;

  M=A->shape[1];

  b->reshape_({1,M});
  Tensor::mult2D(ones,0,b,0,mem,0);
  A->Tensor::add(1.0,A,1.0,mem,A,0);
  b->reshape_({M});

}

void rdiff(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem)
{
  int N,M;

  M=A->shape[1];

  b->reshape_({1,M});
  Tensor::mult2D(ones,0,b,0,mem,0);
  A->Tensor::add(1.0,A,-1.0,mem,A,0);
  b->reshape_({M});

}

void rmult(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem)
{
  int N,M;

  M=A->shape[1];


  b->reshape_({1,M});
  Tensor::mult2D(ones,0,b,0,mem,0);
  Tensor::el_mult(A,mem,A,0);
  b->reshape_({M});

}

void rdiv(Tensor *A, Tensor *b, Tensor *ones,Tensor *mem)
{
  int N,M;

  M=A->shape[1];


  b->reshape_({1,M});
  Tensor::mult2D(ones,0,b,0,mem,0);
  Tensor::el_div(A,mem,A,0);
  b->reshape_({M});

}

void cmean(Tensor *A, Tensor *b,Tensor *ones)
{
  int N,M;

  N=A->shape[0];
  M=A->shape[1];

  ones->reshape_({1,N});
  b->reshape_({1,M});
  Tensor::mult2D(ones,0,A,0,b,0);
  b->div_(N);
  b->reshape_({M});
  ones->reshape_({N,1});
}


void BN_forward(Tensor *input,Tensor *output,  Tensor *bn_mean, Tensor *bn_var, Tensor *mean, Tensor *variance,float momentum, float epsilon, bool affine, Tensor *bn_g, Tensor *bn_b,Tensor *opa, int trmode)
{
  // 2D or 4D batch norm
  // Input = Output = opa = {Batch,Channels,H,W} OR {Batch,Dim}
  // bn_mean = bn_var = mean = variance = bn_g = bn_b = {Channels} or {Dim}

  int M,N;
  int b,z,r,c,d;

  Tensor *in;

  // Permute 4D tensors and set N,M values.
  // Essentialy 4D Tensors are reshaped as 2D and
  // all the batchnorm works over 2D Tensors
  if (input->ndim==2) {
    N=b=input->shape[0];
    M=d=input->shape[1];
    in=input->clone();
  }
  else {
    b=input->shape[0];
    M=z=input->shape[1];
    r=input->shape[2];
    c=input->shape[3];
    N=b*r*c;

    in=new Tensor({b,r,c,z},input->device);
    permute_channels_last(input,in);
    in->reshape_({N,M}); // now is a 2D tensor
  }


  Tensor *var=new Tensor({N,M},input->device);
  Tensor *ones=new Tensor({N,1},input->device);
  ones->fill_(1.0);

  if (trmode) {
    // mean
    cmean(in,bn_mean,ones);

    // in=in-mean
    rdiff(in,bn_mean,ones,var);

    Tensor::copy(in,var);

    // variance
    var->sqr_();
    cmean(var,bn_var,ones);

    // Update global statistics
    Tensor::add(momentum, mean, (1.0-momentum), bn_mean,mean,0);
    Tensor::add(momentum, variance, (1.0-momentum), bn_var,variance,0);

    // sd=sqrt(var+epsilon)
    bn_var->add_(epsilon);
    bn_var->sqrt_();

    // in/sd
    rdiv(in,bn_var,ones,var); //in=(x-mean)/sd
  }
  else {
    rdiff(in,mean,ones,var);
    Tensor::copy(variance,bn_var);
    bn_var->add_(epsilon);
    bn_var->sqrt_();
    rdiv(in,bn_var,ones,var);
  }

  if (affine) {
    // opa: output pre-affice needed for backward
    if (trmode) {
      if (input->ndim==4) opa->reshape_({N,M});
      Tensor::copy(in,opa);
    }
    // apply affine transform in=gamma*in+beta
    rmult(in,bn_g,ones,var);
    rsum(in,bn_b,ones,var);
  }

  // copy in to ouput
  if (input->ndim==4) {permute_channels_first(in,output);}
  else Tensor::copy(in,output);

  // Free
  delete var;
  delete in;
  delete ones;


}

void BN_backward(Tensor* input, Tensor *delta,Tensor *pdelta, Tensor *bn_mean, Tensor *bn_var, Tensor *mean, Tensor *variance,float epsilon, bool affine, Tensor *bn_g, Tensor *bn_b, Tensor *gbn_g, Tensor* gbn_b,Tensor *opa)
{
  int M,N;
  int b,z,r,c,d;

  Tensor *dp;
  Tensor *in;

  if (input->ndim==2) {
    N=b=input->shape[0];
    M=d=input->shape[1];


    dp=delta->clone();
    in=input->clone();
  }
  else {
    b=input->shape[0];
    M=z=input->shape[1];
    r=input->shape[2];
    c=input->shape[3];

    N=b*r*c;

    // permute input and delta
    in=new Tensor({b,r,c,z},input->device);
    dp=new Tensor({b,r,c,z},input->device);

    permute_channels_last(delta,dp);
    permute_channels_last(input,in);

    dp->reshape_({N,M});
    in->reshape_({N,M});

  }

  Tensor *A=new Tensor({N,M},input->device);
  Tensor *ones=new Tensor({1,N},in->device);
  ones->fill_(1.0);
  Tensor *m=new Tensor({1,M},in->device);


  // Affine
  if (affine) {
    //1 gamma
    Tensor::el_mult(dp,opa,opa,0);
    cmean(opa,m,ones);
    Tensor::add(1,gbn_g,1,m,gbn_g,0);

    //2 Beta
    cmean(dp,m,ones);
    Tensor::add(1,gbn_b,1,m,gbn_b,0);

    // Y = OPA
    // dp=dE/dY
    // Obtain dE/dY from delta:
    rmult(dp,bn_g,ones,A);

  }

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
  Tensor::el_mult(dp,opa,A,0);

  //2
  cmean(A,m,ones);

  //3
  rmult(opa,m,ones,A);

  //4
  cmean(dp,m,ones);

  //5
  rsum(opa,m,ones,A);

  // 6
  Tensor::add(1,dp,-1,opa,dp,0);

  // from forward bn_var=sqrt(var(X) + eps
  // 7
  rdiv(dp,bn_var,ones,A);

  // Inc parent delta
  if (input->ndim==4) {
    permute_channels_first(dp,delta);
    Tensor::inc(delta, pdelta);
  }
  else Tensor::inc(dp, pdelta);

  delete ones;
  delete m;
  delete A;
  delete in;
  delete dp;


}
