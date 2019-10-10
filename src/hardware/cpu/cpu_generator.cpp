//
// Created by Salva Carrión on 30/09/2019.
//

#include "cpu_hw.h"
#include "../../random.h"

void cpu_rand_uniform(Tensor *A, float v) {
    for (int i = 0; i < A->size; ++i) A->ptr[i] = uniform() * v;
}

void cpu_rand_signed_uniform(Tensor *A, float v){
    for (int i = 0; i < A->size; ++i) A->ptr[i] = signed_uniform() * v;
}

void cpu_rand_binary(Tensor *A, float v){
    for (int i = 0; i < A->size; ++i)
        if (uniform() < v) A->ptr[i] = 1.0;
        else A->ptr[i] = 0.0;
}

void cpu_rand_normal(Tensor *A, float m, float s, bool fast_math){
    int r=rand();
    if(fast_math){
        for (int i = 0; i < A->size; ++i) A->ptr[i] = fast_randn(m, s, r++);
    }else{
        for (int i = 0; i < A->size; ++i) A->ptr[i] = slow_randn(m, s);
    }
}
