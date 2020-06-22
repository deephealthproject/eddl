/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_tensor.h"

void cpu_range(Tensor *A, float min, float step){
    _profile(_CPU_RANGE, 0);
    float v=min;

    //#pragma omp parallel for
    for(int i=0; i<A->size; i++){
        A->ptr[i] = v;
        v+=step;
    }
    _profile(_CPU_RANGE, 1);
}

void cpu_eye(Tensor *A, int offset){
    _profile(_CPU_EYE, 0);
    #pragma omp parallel for
    for(int i=0; i<A->size; i++){
        if ((i/A->shape[0]+offset) == i%A->shape[1]){ A->ptr[i] = 1.0f; }  // rows+offset == col?
        else { A->ptr[i] = 0.0f; }
    }
    _profile(_CPU_EYE, 1);
}

void cpu_diag(Tensor *A, Tensor *B, int k){
    #pragma omp parallel for
    for(int i=0; i<A->size; i++){
        if ((i/A->shape[0]+k) == i%A->shape[1]){ B->ptr[i] = A->ptr[i]; }  // rows+offset == col?
        else { B->ptr[i] = 0.0f; }
    }
}