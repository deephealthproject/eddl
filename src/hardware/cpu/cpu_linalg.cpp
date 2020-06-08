/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/



#include <iostream>
#include <utility>
#include <cmath>

#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/random.h"

float cpu_norm(Tensor *A, string ord){
    float norm = 0.0f;

    if(ord=="fro"){

        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < A->size; ++i){
            norm += A->ptr[i]*A->ptr[i];
        }

    }else{
       msg("Not yet implemented", "cpu_norm");
    }

    return sqrt(norm);
}