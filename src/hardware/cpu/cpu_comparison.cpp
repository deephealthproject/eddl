//
// Created by Salva Carrión on 30/09/2019.
//

#include "cpu_hw.h"

int cpu_equal(Tensor *A, Tensor *B){
    for (int i = 0; i < A->size; i++)
        if (fabs(A->ptr[i]-B->ptr[i])>0.001) {
            fprintf(stderr,"\n>>>>>>>>>>\n");
            fprintf(stderr,"%f != %f\n",A->ptr[i],B->ptr[i]);
            return 0;
        }
}