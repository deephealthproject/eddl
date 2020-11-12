/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/system_info.h"
#include <limits>

// CPU: Logic functions: Truth value testing
bool cpu_all(Tensor *A){
    bool res = true;

    _profile(_CPU_ALL, 0);

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if (A->ptr[i] != 1.0f){
            #pragma omp critical
            {
                res = false;
            }
#if defined(EDDL_LINUX) || defined(EDDL_UNIX) || defined(EDDL_APPLE)
            #pragma omp cancel for
#endif
        }
    }
    _profile(_CPU_ALL, 1);
    return res;
}

bool cpu_any(Tensor *A){
    bool res = false;

    _profile(_CPU_ANY, 0);


    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if (A->ptr[i] == 1.0f){
            #pragma omp critical
            {
                res = true;
            }
#if defined(EDDL_LINUX) || defined(EDDL_UNIX) || defined(EDDL_APPLE)
            #pragma omp cancel for
#endif
        }
    }
    _profile(_CPU_ANY, 1);
    return res;
}

// CPU: Logic functions: Comparisons
void cpu_isfinite(Tensor *A, Tensor* B){
    _profile(_CPU_ISFINITE, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isfinite(A->ptr[i]);
    }
    _profile(_CPU_ISFINITE, 1);
}

void cpu_isinf(Tensor *A, Tensor* B){
    _profile(_CPU_ISINF, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isinf(A->ptr[i]);
    }
    _profile(_CPU_ISINF, 1);
}

void cpu_isnan(Tensor *A, Tensor* B){
    _profile(_CPU_ISNAN, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isnan(A->ptr[i]);
    }
    _profile(_CPU_ISNAN, 1);
}


void cpu_isneginf(Tensor *A, Tensor* B){
    _profile(_CPU_ISNEGINF, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isinf(A->ptr[i]) && A->ptr[i] < 0.0f;
    }
    _profile(_CPU_ISNEGINF, 1);
}

void cpu_isposinf(Tensor *A, Tensor* B){
    _profile(_CPU_ISPOSINF, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = std::isinf(A->ptr[i]) && A->ptr[i] > 0.0f;
    }
    _profile(_CPU_ISPOSINF, 1);
}


// CPU: Logic functions: Comparisons
void cpu_logical_and(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_LOGICAL_AND, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = (bool)A->ptr[i] & (bool)B->ptr[i];
    }
    _profile(_CPU_LOGICAL_AND, 1);
}

void cpu_logical_or(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_LOGICAL_OR, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = (bool)A->ptr[i] | (bool)B->ptr[i];
    }
    _profile(_CPU_LOGICAL_OR, 1);
}

void cpu_logical_not(Tensor *A, Tensor *B){
    _profile(_CPU_LOGICAL_NOT, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = !((bool)A->ptr[i]);  // why not use "~"
    }
    _profile(_CPU_LOGICAL_NOT, 1);
}

void cpu_logical_xor(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_LOGICAL_XOR, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = (bool)A->ptr[i] ^ (bool)B->ptr[i];
    }
    _profile(_CPU_LOGICAL_XOR, 1);
}


// CPU: Logic functions: Comparisons

bool cpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    bool allclose = true;
    int first_idx = -1;

    _profile(_CPU_ALLCLOSE, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        bool close = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
        if (!close){
            #pragma omp critical
            {
                allclose = false;
                if(first_idx < 0) { first_idx=i; }

            }
#if defined(EDDL_LINUX) || defined(EDDL_UNIX) || defined(EDDL_APPLE)
            #pragma omp cancel for
#endif
        }
    }
    _profile(_CPU_ALLCLOSE, 1);

//    // TODO: temp!
//    if(!allclose){
//        fprintf(stderr, "\n>>>>>>>>>>\n");
//        fprintf(stderr, "[values]\t\t%f != %f\n", A->ptr[first_idx], B->ptr[first_idx]);
//        fprintf(stderr, "[diff=\t%f (rtol=%f; atol=%f)]\n", A->ptr[first_idx] - B->ptr[first_idx], rtol, atol);
//        fprintf(stderr, "<<<<<<<<<<\n");
//    }

    return allclose;
}

void cpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    _profile(_CPU_ISCLOSE, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = ::fabsf(A->ptr[i] - B->ptr[i]) <= (atol + rtol * ::fabsf(B->ptr[i]));
    }
    _profile(_CPU_ISCLOSE, 1);
}


void cpu_greater(Tensor *A, Tensor *B, float v){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = A->ptr[i] > v;
    }
}

void cpu_greater(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_GREATER, 0);

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] > B->ptr[i];
    }
    _profile(_CPU_GREATER, 1);
}


void cpu_greater_equal(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = A->ptr[i] >= v;
    }
}

void cpu_greater_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_GREATER_EQUAL, 0);

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] >= B->ptr[i];
    }
    _profile(_CPU_GREATER_EQUAL, 1);
}

void cpu_less(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = A->ptr[i] < v;
    }
}

void cpu_less(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_LESS, 0);

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] < B->ptr[i];
    }
    _profile(_CPU_LESS, 1);
}

void cpu_less_equal(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = A->ptr[i] <= v;
    }
}

void cpu_less_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_LESS_EQUAL, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] <= B->ptr[i];
    }
    _profile(_CPU_LESS_EQUAL, 1);
}

void cpu_equal(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = A->ptr[i] == v;
    }
}

void cpu_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_EQUAL, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] == B->ptr[i];
    }
    _profile(_CPU_EQUAL, 1);
}

void cpu_not_equal(Tensor *A, Tensor *B, float v){
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        B->ptr[i] = A->ptr[i] != v;
    }
}

void cpu_not_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile(_CPU_NOT_EQUAL, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        C->ptr[i] = A->ptr[i] != B->ptr[i];
    }
    _profile(_CPU_NOT_EQUAL, 1);
}



int cpu_equal2(Tensor *A, Tensor *B, float epsilon){
  _profile(_CPU_EQUAL2, 0);

  for (int i = 0; i < A->size; i++){
      float delta = ::fabs(A->ptr[i] - B->ptr[i]);
      if (delta > epsilon) {
          fprintf(stderr, "\n>>>>>>>>>>\n");
          fprintf(stderr, "[values]\t\t%f != %f\n", A->ptr[i], B->ptr[i]);
          fprintf(stderr, "[diff/epsilon]\t%f > %f\n", delta, epsilon);
          fprintf(stderr, "<<<<<<<<<<\n");
          _profile(_CPU_EQUAL2, 1);
          return 0;
      }
  }
  _profile(_CPU_EQUAL2, 1);
  return 1;
}
