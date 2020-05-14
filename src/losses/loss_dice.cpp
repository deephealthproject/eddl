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

float LDice::value(Tensor *T, Tensor *Y) {
  //2*sum(|A*B|)/(sum(A^2)+sum(B^2))
  Tensor *A;
  Tensor *B;
  Tensor *C;

  float D;

  A=T->clone();
  B=Y->clone();


  C=A->clone();


  Tensor::el_mult(A,B,C,0);
  C->abs_();
  float n=2*C->sum();


  A->sqr_();
  float sA=A->sum();

  B->sqr_();
  float sB=B->sum();

  D=n/(sA+sB);

  delete A;
  delete B;
  delete C;

  return D*T->shape[0]; // batch is divided in print_loss
}

Loss* LDice::clone()
{
  return new LDice();
}
