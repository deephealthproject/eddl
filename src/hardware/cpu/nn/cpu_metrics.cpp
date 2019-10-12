/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "cpu_nn.h"

int cpu_accuracy(Tensor *A, Tensor *B){
  int acc = 0;
  int aind, bind;

  #pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
    (*A->ptr2).col(i).maxCoeff(&aind);
    (*B->ptr2).col(i).maxCoeff(&bind);
    if (aind == bind) acc++;
  }
  return acc;
}
