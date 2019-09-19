#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "../utils.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;

int Tensor::accuracy(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
    if (!eqsize(A, B)) msg("Incompatible dims", "Tensor::accuracy");
    if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x probs)", "Tensor::Accuracy");

    int acc = 0;

    B->tsem->lock();

    if (A->isCPU()) {
        int aind, bind;

        for (int i = 0; i < A->shape[0]; i++) {
            (*A->ptr2).col(i).maxCoeff(&aind);
            (*B->ptr2).col(i).maxCoeff(&bind);
            if (aind == bind) acc++;
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_accuracy(A,B,&acc);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    B->tsem->unlock();
    return acc;

}