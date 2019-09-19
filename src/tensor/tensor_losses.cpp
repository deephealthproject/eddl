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

// Cross-Entropy: C=-(A*log(B)+(1-A)*log(1-B))
void Tensor::cent(Tensor *A, Tensor *B, Tensor *C) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::cross-entropy");
    if ((!eqsize(A, B)) || (!eqsize(A, C))) msg("Incompatible dims", "Tensor::cross-entropy");

    C->tsem->lock();
    if (A->isCPU()) {

        for (int i = 0; i < A->size; i++) {
            C->ptr[i] = 0;
            if (A->ptr[i] != 0.0) C->ptr[i] -= A->ptr[i] * std::log(B->ptr[i]+0.00001);
            if (A->ptr[i] != 1.0) C->ptr[i] -= (1.0 - A->ptr[i]) * std::log(1.0 - B->ptr[i]+0.00001);
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
         gpu_cent(A,B,C);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    C->tsem->unlock();
}
