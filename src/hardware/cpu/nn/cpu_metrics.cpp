/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "hardware/cpu/nn/cpu_nn.h"

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
