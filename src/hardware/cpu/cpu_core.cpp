//
// Created by Salva Carrión on 30/09/2019.
//

#include "cpu_hw.h"

void cpu_transpose(Tensor *A, Tensor *B) {
    for (int i = 0; i < A->size; i++)
        B->ptr[i] = A->ptr[i];
}

void cpu_copy(Tensor *A, Tensor *B){
    for (int i = 0; i < A->size; i++)
        B->ptr[i] = A->ptr[i];
}

void cpu_fill(Tensor *A, int aini, int aend, Tensor *B, int bini, int bend, int inc){
    int at = A->size / A->shape[0];
    int bt = B->size / B->shape[0];

    int t = 1;
    for (int i = 2; i < A->ndim; i++)
        t *= A->shape[i];

    for (int i = 0; i < A->shape[0]; i++) {
        int ap = (i * at) + (aini * t);
        int bp = (i * bt) + (bini * t);

        for (int j = aini; j < aend; j++) {
            for (int k = 0; k < t; k++, ap++, bp++)
                if (inc) B->ptr[bp] += A->ptr[ap];
                else B->ptr[bp] = A->ptr[ap];
        }
    }
}

void cpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end){
    int s = A->size / A->shape[0];

    for (int i = ini; i < end; i++) {
        int p = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            B->ptr[pb] = A->ptr[p];
    }
}
