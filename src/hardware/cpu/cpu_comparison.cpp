/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_hw.h"
#include <limits>

// CPU: Logic functions: Truth value testing
bool cpu_all(Tensor *A){
    bool res = true;

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if (A->ptr[i] != 1.0f){
            #pragma omp critical
            {
                res = false;
            }
#if OpenMP_VERSION_MAJOR >= 4
            #pragma omp cancel for
#endif // OpenMP_VERSION_MAJOR >= 4
        }
    }
    return res;
}

bool cpu_any(Tensor *A){
    bool res = false;

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if (A->ptr[i] == 1.0f){
            #pragma omp critical
            {
                res = true;
            }
#if OpenMP_VERSION_MAJOR >= 4
            #pragma omp cancel for
#endif // OpenMP_VERSION_MAJOR >= 4
        }
    }
    return res;
}

// CPU: Logic functions: Comparisons
void cpu_isfinite(Tensor *A, Tensor* B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isfinite(A->ptr[i]);
    }
}

void cpu_isinf(Tensor *A, Tensor* B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isinf(A->ptr[i]);
    }
}

void cpu_isnan(Tensor *A, Tensor* B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isnan(A->ptr[i]);
    }
}


void cpu_isneginf(Tensor *A, Tensor* B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isinf(A->ptr[i]) && A->ptr[i] < 0.0f;
    }
}

void cpu_isposinf(Tensor *A, Tensor* B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isinf(A->ptr[i]) && A->ptr[i] > 0.0f;
    }
}


// CPU: Logic functions: Comparisons
void cpu_logical_and(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = (bool)A->ptr[i] & (bool)B->ptr[i];
    }
}

void cpu_logical_or(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = (bool)A->ptr[i] | (bool)B->ptr[i];
    }
}

void cpu_logical_not(Tensor *A, Tensor *B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = !((bool)A->ptr[i]);  // why not use "~"
    }
}

void cpu_logical_xor(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = (bool)A->ptr[i] ^ (bool)B->ptr[i];
    }
}


// CPU: Logic functions: Comparisons

bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    bool allclose = true;

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        bool close = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
        if (!close){
            #pragma omp critical
            {
                allclose = false;
            }
#if OpenMP_VERSION_MAJOR >= 4
            #pragma omp cancel for
#endif // OpenMP_VERSION_MAJOR >= 4
        }
    }
    return allclose;
}

void cpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
    }
}

void cpu_greater(Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] > B->ptr[i];
    }
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
