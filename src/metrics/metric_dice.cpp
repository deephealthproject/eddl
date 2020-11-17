/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/metrics/metric.h"

using namespace std;


MDice::MDice() : Metric("dice"){}

// TODO: general implementation in tensor reduction
// {nxd} --> {nxd}
void reduced_dice_sum_keep(Tensor * input, Tensor *output,int inc)
{
  int n,d;
  n=input->shape[0];
  d=input->shape[1];

  // nxd
  Tensor *A=input->clone();

  // dx1
  Tensor *ones=new Tensor({d,1},input->device);
  ones->fill_(1.0);

  // nx1
  Tensor *red=new Tensor({n,1},input->device);

  // {nxd} x {dx1} --> {nx1}
  Tensor::mult2D(A,0,ones,0,red,0);

  // 1xd
  ones->reshape_({1,d});

  // {nx1} x {1xd} --> {nxd}
  Tensor::mult2D(red,0,ones,0,output,inc);


  delete A;
  delete ones;
  delete red;

}

float MDice::value(Tensor *T, Tensor *Y) {
  //2*sum(A*B)/(sum(A)+sum(B))
  //2*sum(T*Y)/(sum(T)+sum(Y))
  Tensor *A;
  Tensor *B;
  Tensor *Num;
  Tensor *Den;

  int b=T->shape[0];
  int d=T->size/T->shape[0];


  A=T->clone();
  A->reshape_({b,d});
  B=Y->clone();
  B->reshape_({b,d});

  Num=new Tensor(A->shape,A->device);
  Den=new Tensor(A->shape,A->device);


  // (sum(T)+sum(Y))
  reduced_dice_sum_keep(A,Den,0);
  reduced_dice_sum_keep(B,Den,1);


  // 2*sum(A*B)
  Tensor::el_mult(A,B,A,0);
  reduced_dice_sum_keep(A,Num,0);
  Num->mult_(2.0);

  // 2*sum(A*B)/(sum(T)+sum(Y))
  Tensor::el_div(Num,Den,Den,0);

  float n=Den->sum()/d;

  delete A;
  delete B;
  delete Num;
  delete Den;


  return n;
}

Metric* MDice::clone() {
  return new MDice();
}
