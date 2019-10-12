/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "cpu_hw.h"

int cpu_equal(Tensor *A, Tensor *B){

  for (int i = 0; i < A->size; i++)
  if (::fabs(A->ptr[i]-B->ptr[i])>0.001) {
    fprintf(stderr,"\n>>>>>>>>>>\n");
    fprintf(stderr,"%f != %f\n",A->ptr[i], B->ptr[i]);
    return 0;
  }
}
