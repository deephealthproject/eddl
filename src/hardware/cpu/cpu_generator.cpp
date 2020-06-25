/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/random.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

void cpu_rand_uniform(Tensor * A, float v)
{
    _profile(_CPU_RAND_UNIFORM, 0);
#pragma omp parallel for
    for (int i = 0; i < A->size; ++i) A->ptr[i] = uniform() * v;
    _profile(_CPU_RAND_UNIFORM, 1);
}

void cpu_rand_signed_uniform(Tensor * A, float v)
{
    _profile(_CPU_RAND_SIGNED_UNIFORM, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i) A->ptr[i] = signed_uniform() * v;
    _profile(_CPU_RAND_SIGNED_UNIFORM, 1);
}

void cpu_rand_binary(Tensor * A, float v)
{
    _profile(_CPU_BINARY, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i)
        if (uniform() < v) A->ptr[i] = 1.0;
        else A->ptr[i] = 0.0;
    _profile(_CPU_BINARY, 1);
}

void cpu_rand_normal(Tensor * A, float m, float s, bool fast_math) {
    _profile(_CPU_RAND_NORMAL, 0);
    int r = rand();

    if (fast_math) {
        for (int i = 0; i < A->size; ++i) A->ptr[i] = fast_randn(m, s, r++);
    } else  {
        for (int i = 0; i < A->size; ++i) A->ptr[i] = slow_randn(m, s);
    }
    _profile(_CPU_RAND_NORMAL, 0);
}
