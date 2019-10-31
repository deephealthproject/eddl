/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_kernels.h"


// GPU: Math (in-place)
__global__ void shift(float* a, float* b, int batch, int depth, int irows, int icols, int* shift, int mode, float constant){
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    if (thread_id_x < size){
        int A_stride[4] = {depth*irows*icols, irows*icols, icols, 1};
        int B_stride[4] = A_stride;

        //--------------
        int b = thread_id_x / this->stride[0] % batch;
        int c = thread_id_x / this->stride[1] % depth;
        int Bi = thread_id_x / this->stride[2] % irows;
        int Bj = thread_id_x / this->stride[3] % icols;
        //--------------

        int Ai = Bi - shift[0];
        int Aj = Bj - shift[1];

        if (Ai >= 0 && Ai < irows && Aj >= 0 && Aj < icols){
            int A_pos = b*A_stride[0] + c*A_stride[1] + Ai*A_stride[2] + Aj*A_stride[3];
            int B_pos = b*B_stride[0] + c*B_stride[1] + Bi*B_stride[2] + Bj*B_stride[3];
            B->ptr[B_pos] = A->ptr[A_pos];
        }
    }

}