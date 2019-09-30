//
// Created by Salva Carrión on 30/09/2019.
//

#include "cpu_hw.h"

void cpu_reduce_sum2D(Tensor *A, Tensor *B, int axis, int incB) {
    if (axis == 0) {
        if (!incB) for (int i = 0; i < A->shape[1]; ++i) B->ptr[i] = 0;

        int p = 0;
        for (int i = 0; i < A->shape[0]; ++i) {
            for (int j = 0; j < A->shape[1]; ++j, p++)
                B->ptr[j] += A->ptr[p];
        }

    } else {
        if (!incB) for (int i = 0; i < A->shape[0]; ++i) B->ptr[i] = 0;

        int p = 0;
        for (int i = 0; i < A->shape[0]; ++i) {
            for (int j = 0; j < A->shape[1]; ++j, p++)
                B->ptr[i] += A->ptr[p];
        }
    }
}

void cpu_reduceTosum(Tensor *A, Tensor *B, int axis){
    for (int i = 0; i < B->size; i++)
        for (int j = 0; j < A->shape[axis]; j++)
            B->ptr[i] += A->ptr[j];
}