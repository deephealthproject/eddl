/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "cpu_hw.h"

int cpu_equal(Tensor *A, Tensor *B, float epsilon){

  for (int i = 0; i < A->size; i++){
      float delta = ::fabs(A->ptr[i] - B->ptr[i]);
      if (delta > epsilon) {
          fprintf(stderr, "\n>>>>>>>>>>\n");
          fprintf(stderr, "%f != %f\n", A->ptr[i], B->ptr[i]);
          fprintf(stderr, "%f > %f\n", delta, epsilon);
          return 0;
      }
  }
    return 1;
}
