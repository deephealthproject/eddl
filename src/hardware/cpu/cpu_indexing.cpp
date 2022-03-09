/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_tensor.h"



std::pair<unsigned int*, int> cpu_nonzero(Tensor *A){
    // This can be improved:
    // See: https://stackoverflow.com/questions/18971401/sparse-array-compression-using-simd-avx2/41958528#41958528
    auto* indices = new unsigned int[A->size];
    unsigned int size = 0;

    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){

        if (A->ptr[i] != 0.0f){
            #pragma omp critical
            {
                indices[size++] = i;
            }
        }
    }

    // Copy data
    auto* new_data = new unsigned int[size];
    std::copy(indices, indices + size, new_data);
    delete[] indices;

    return std::make_pair(new_data, size);
}


void cpu_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C){
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        if((bool) condition->ptr[i]){
            C->ptr[i] = A->ptr[i];
        }else{
            C->ptr[i] = B->ptr[i];
        }
    }
}

void cpu_where_back(Tensor *condition, Tensor *PD_A, Tensor *PD_B, Tensor *D){
#pragma omp parallel for
    for (int i = 0; i < PD_A->size; ++i){
        if((bool) condition->ptr[i]){
            PD_A->ptr[i] += D->ptr[i];
        }else{
            PD_B->ptr[i] += D->ptr[i];
        }
    }
}