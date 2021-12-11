/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <iostream>
#include <utility>
#include <cmath>

#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/random.h"



float cpu_norm(Tensor *A, string ord){
    return cpu_norm_(A->ptr, A->size, nullptr, ord);
}


void cpu_norm(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, string ord){
#pragma omp parallel for
    for(int i=0; i<rd->index.size(); i++){
        B->ptr[i] = cpu_norm_(A->ptr, rd->index[i].size(), rd->index[i].data(), ord);
    }
}

float cpu_norm_(float *ptr, int size, int *map, string ord){
    float norm = 0.0f;

    if(ord=="fro"){

        // TODO: I don't like this approach
        if(map == nullptr){
            #pragma omp parallel for reduction(+:norm)
            for (int i = 0; i < size; ++i){ norm += ::pow(ptr[i], 2); }  // Compiler trick: pow(x,2) == x*x
        }else{
            #pragma omp parallel for reduction(+:norm)
            for (int i = 0; i < size; ++i){ norm += ::pow(ptr[map[i]], 2); }
        }

    }else{
        msg("Not yet implemented", "cpu_norm");
    }

    return ::sqrtf(norm);
}

