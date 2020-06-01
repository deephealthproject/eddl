/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include "eddl/hardware/cpu/nn/cpu_nn.h"

void cpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
    #ifdef DEBUG_FUNCS 
    printf("cpu_repeat_nn\n");
    #endif
    // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)
    #pragma omp parallel for
    for(int i=0; i<B->size; i++){
        // Get row/col of Tensor B
        int row_b = i/B->shape[2+1];  // (batch, channels, rows), cols
        int col_b = i%B->shape[2+1]; // (batch, channels, rows), cols

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_b/size[0];
        int col_a = col_b/size[1];
        int offset_a = row_a*A->shape[2+1] + col_a;

        B->ptr[i] = A->ptr[offset_a];
    }

}

void cpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    #ifdef DEBUG_FUNCS 
    printf("cpu_d_repeat_nn\n");
    #endif
    // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)
    ////#pragma omp parallel for

    #pragma omp parallel for
    for(int i=0; i<D->size; i++){
        // Get row/col of Tensor B
        int row_d = i/D->shape[2+1];  // (batch, channels, rows), cols
        int col_d = i%D->shape[2+1];  // (batch, channels, rows), cols

        // Translate row/col of Tensor B to Tensor A
        int row_a = row_d/size[0];
        int col_a = col_d/size[1];
        int offset_a = row_a*A->shape[2+1] + col_a;

        A->ptr[offset_a] += D->ptr[i];
    }

}
