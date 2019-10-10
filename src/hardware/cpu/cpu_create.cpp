/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "cpu_hw.h"

void cpu_range(Tensor *A, float min, float step){
    float v=min;
    for(int i=0; i<A->size; i++){
        A->ptr[i] = v;
        v+=step;
    }
}
