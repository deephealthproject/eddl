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


void cpu_cent(Tensor *A, Tensor *B, Tensor *C){
  #pragma omp parallel for
  for (int i = 0; i < A->size; i++) {
    C->ptr[i] = 0;
    if (A->ptr[i] != 0.0) C->ptr[i] -= A->ptr[i] * std::log(B->ptr[i]+0.00001);
    if (A->ptr[i] != 1.0) C->ptr[i] -= (1.0 - A->ptr[i]) * std::log(1.0 - B->ptr[i]+0.00001);
  }
}
