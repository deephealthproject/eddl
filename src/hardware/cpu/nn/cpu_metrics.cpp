/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

int cpu_accuracy(Tensor *A, Tensor *B){
  int acc = 0;
  int aind, bind;

  for (int i = 0; i < A->shape[0]; i++) {
    (*A->ptr2).col(i).maxCoeff(&aind);
    (*B->ptr2).col(i).maxCoeff(&bind);
    if (aind == bind) acc++;
  }
  return acc;
}

int cpu_bin_accuracy(Tensor *A, Tensor *B){
  int acc = 0;

  for (int i = 0; i < A->shape[0]; i++)
    if ((B->ptr[i]>0.5)&&(A->ptr[i]==1.0)) acc++;
    else if ((B->ptr[i]<=0.5)&&(A->ptr[i]==0.0)) acc++;

  return acc;
}
