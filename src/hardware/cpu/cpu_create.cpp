//
// Created by Salva Carrión on 30/09/2019.
//

#include "cpu_hw.h"

void cpu_range(Tensor *A, float min, float step){
    float v=min;
    for(int i=0; i<A->size; i++){
        A->ptr[i] = v;
        v+=step;
    }
}
