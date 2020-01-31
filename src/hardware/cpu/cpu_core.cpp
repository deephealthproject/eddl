/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 0.3
 * copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
 * Date: October 2019
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */


#include "cpu_hw.h"

void cpu_transpose(Tensor * A, Tensor * B) {
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++)
        B->ptr[i] = A->ptr[i];
}

void cpu_copy(Tensor * A, Tensor * B){
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++)
        B->ptr[i] = A->ptr[i];
}

void cpu_fill_(Tensor *A, float v){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i)
        A->ptr[i] = v;
}

void cpu_fill(Tensor * A, int aini, int aend, Tensor * B, int bini, int bend, int inc){
    int at = A->size / A->shape[0];
    int bt = B->size / B->shape[0];

    int t = 1;


    for (int i = 2; i < A->ndim; i++)
        t *= A->shape[i];

    #pragma omp parallel for
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


void cpu_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    #pragma omp parallel for
    for (int i = 0; i < B->size; i++) {
        B->ptr[i] = A->ptr[sd->cpu_addresses[i]];
    }
}

void cpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    #pragma omp parallel for
    for (int i = 0; i < A->size; i++) {  // walk stride
        B->ptr[sd->cpu_addresses[i]] += A->ptr[i];  // delta_parent += delta
    }
}

void cpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    #pragma omp parallel for
    for (int i = 0; i < B->size; i++) {
        A->ptr[sd->cpu_addresses[i]] = B->ptr[i];
    }
}
void cpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    #pragma omp parallel for
    for (int i = 0; i < B->size; i++) {
        B->ptr[i] += A->ptr[sd->cpu_addresses[i]];
    }
}


void cpu_select(Tensor * A, Tensor * B, vector<int> sind, int ini, int end)
{
    int s = A->size / A->shape[0];


    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            B->ptr[pb] = A->ptr[p];
    }
}

void cpu_deselect(Tensor * A, Tensor * B, vector<int> sind, int ini, int end){
    int s = A->size / A->shape[0];

    #pragma omp parallel for
    for (int i = ini; i < end; i++) {
        int p  = sind[i] * s;
        int pb = (i - ini) * s;
        for (int j = 0; j < s; j++, p++, pb++)
            B->ptr[p] = A->ptr[pb];
    }
}

void cpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative){

    // Initial offsets
    unsigned int dest_offset = 0;
    unsigned int *src_offset = new unsigned int[t.size()]();  // zeros

    // If we haven't walked throught the entire new_tensor, keep walking
    while(dest_offset < A->size) {

        // Walk through all the tensors to concat one axis (once)
        for (unsigned int i = 0; i < t.size(); i++) {
            unsigned int size = t[i]->stride[axis] * t[i]->shape[axis];

            // Copy n bytes from src to dest
            float *dest = A->ptr + dest_offset;
            float *src = t[i]->ptr + src_offset[i];

            if(derivative){
                for (unsigned int k = 0; k < size; k++) {
                    *(src+k) += *(dest+k);
                }
            }else{
                memcpy(dest, src, sizeof(float) * size);
            }

            // Step stride
            dest_offset += size;
            src_offset[i] += size;
        }
    }

}

