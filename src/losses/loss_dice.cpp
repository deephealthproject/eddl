/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/losses/loss.h"

using namespace std;


LDice::LDice() : Loss("dice"){}

// {nxd} --> {nx1}
void reduced_sum_keep(Tensor * input, Tensor *output,int inc)
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
/*

// reduction batch
void LDice::delta(Tensor *T, Tensor *Y, Tensor *D) {
    //delta: 2*[sum(T)*(sum(Y)+sum(T))-Dim*sum(T*Y)]/(sum(Y)+sum(T)^2)
    Tensor *A;
    Tensor *B;
    Tensor *C;
    int d=T->size/T->shape[0];

    A=T->clone();
    B=Y->clone();
    C=A->clone();

    //(sum(Y)+sum(T))
    reduced_sum_keep(A,D,0);
    reduced_sum_keep(B,D,1);

    // -Dim*sum(T*Y)
    //T*Y
    Tensor::el_mult(A,B,C,0);

    // sum(T*Y)
    reduced_sum_keep(C,C,0);

    // -Dim*sum(T*Y)
    C->mult_(-d);
    ////

    //sum(T)
    reduced_sum_keep(A,A,0);

    //sum(T)*(sum(Y)+sum(T))-Dim*sum(T*Y)
    Tensor::el_mult(A,D,C,1); //inc=1 -> C=C+A*D


    //(sum(Y)+sum(T))^2
    D->sqr_();

    Tensor::el_div(C,D,D,0);

    delete A;
    delete B;
    delete C;

    D->div_(-D->shape[0]); // -1 want to maximize


}
*/


// redcution to batch, image level value
float LDice::value(Tensor *T, Tensor *Y) {
  //2*sum(A*B)/(sum(A)+sum(B))
  //2*sum(T*Y)/(sum(T)+sum(Y))
  Tensor *A;
  Tensor *B;
  Tensor *C;

  float D;

  int d=T->size/T->shape[0];

  A=T->clone();
  B=Y->clone();
  C=A->clone();

  // (sum(T)+sum(Y))
  reduced_sum_keep(A,C,0);
  reduced_sum_keep(B,C,1);


  // 2*sum(A*B)
  Tensor::el_mult(A,B,A,0);
  reduced_sum_keep(A,A,0);
  A->mult_(2.0);

  // 2*sum(A*B)/(sum(T)+sum(Y))
  Tensor::el_div(A,C,C,0);

  float n=C->sum()/d;

  delete A;
  delete B;
  delete C;

  return n;
}


//https://arxiv.org/pdf/1606.04797v1.pdf
void LDice::delta(Tensor *T, Tensor *Y, Tensor *D) {
    //delta: 2*[T*(sum(Y^2)+sum(T^2))-2Y(sum(T*Y))/(sum(Y^2)+sum(T^2))^2]
    Tensor *A;
    Tensor *B;
    Tensor *C;

    A=T->clone();
    B=Y->clone();
    C=A->clone();

    // -2Y(sum(T*Y))
    Tensor::el_mult(A,B,C,0);
    float n=-2*C->sum();
    B->mult_(n);
    ////

    //T*(sum(Y^2)+sum(T^2))
    A->sqr_();
    float sT=A->sum();

    Tensor::copy(Y,C);
    C->sqr_();
    float sY=C->sum();
    float sTY=sT+sY;

    Tensor::copy(T,A);
    A->mult_(sTY);
    ///

    Tensor::add(1.0, A,1.0, B,D,0);
    D->mult_(2.0); // 2*[]

    delete A;
    delete B;
    delete C;

    D->div_(-1*sTY*sTY*D->shape[0]); // -1 want to maximize


}



Loss* LDice::clone()
{
  return new LDice();
}
