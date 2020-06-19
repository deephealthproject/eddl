/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
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

// {nxd} --> {nxd}
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


// reduction batch
void LDice::delta(Tensor *T, Tensor *Y, Tensor *D) {
    //delta: 2*[T*(sum(Y)+sum(T))-sum(T*Y)]/(sum(Y)+sum(T)^2)
    Tensor *A;
    Tensor *B;
    Tensor *C;
    vector<int> shape=D->shape;

    int b=T->shape[0];
    int d=T->size/T->shape[0];

    A=T->clone();
    A->reshape_({b,d});

    B=Y->clone();
    B->reshape_({b,d});

    C=A->clone();
    C->reshape_({b,d});

    D->reshape_({b,d});

    //(sum(Y)+sum(T))
    reduced_sum_keep(A,D,0);
    reduced_sum_keep(B,D,1);


    // -sum(T*Y)
    //T*Y
    Tensor::el_mult(A,B,C,0);

    // sum(T*Y)
    reduced_sum_keep(C,C,0);
    C->mult_(-1.0);
    ////

    //T*(sum(Y)+sum(T))-sum(T*Y)
    Tensor::el_mult(A,D,C,1); //inc=1 -> C=C+A*D

    //(sum(Y)+sum(T))^2
    D->sqr_();

    Tensor::el_div(C,D,D,0);

    delete A;
    delete B;
    delete C;

    D->reshape_(shape);

    D->mult_(-1);//D->shape[0]); // -1 want to maximize


}



// redcution to batch, image level value
float LDice::value(Tensor *T, Tensor *Y) {
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
  reduced_sum_keep(A,Den,0);
  reduced_sum_keep(B,Den,1);


  // 2*sum(A*B)
  Tensor::el_mult(A,B,A,0);
  reduced_sum_keep(A,Num,0);
  Num->mult_(2.0);

  // 2*sum(A*B)/(sum(T)+sum(Y))
  Tensor::el_div(Num,Den,Den,0);

  float n=Den->sum()/d;

  delete A;
  delete B;
  delete Num;
  delete Den;

  return T->shape[0]-n;
}


Loss* LDice::clone()
{
  return new LDice();
}
