/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "cpu_hw.h"

bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        bool close = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
        if (!close){
            return false;
        }
    }
    return true;
}

void cpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
    }
}

void cpu_greater(Tensor *A, Tensor *B, Tensor *C){

}

void cpu_greater_equal(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] >= B->ptr[i];
    }
}

void cpu_less(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] < B->ptr[i];
    }
}

void cpu_less_equal(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] <= B->ptr[i];
    }
}

void cpu_equal(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] == B->ptr[i];
    }
}

void cpu_not_equal(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] != B->ptr[i];
    }
}



int cpu_equal2(Tensor *A, Tensor *B, float epsilon){

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
