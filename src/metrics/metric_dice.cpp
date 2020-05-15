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

#include "eddl/metrics/metric.h"

using namespace std;


MDice::MDice() : Metric("dice"){}

float MDice::value(Tensor *T, Tensor *Y) {
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

Metric* MDice::clone() {
  return new MDice();
}
