/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_nn.h"
#endif


// Cross-Entropy: C=-(A*log(B)+(1-A)*log_(1-B))
void cent(Tensor *A, Tensor *B, Tensor *C) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::cross-entropy");
    if ((!Tensor::eqsize(A, B)) || (!Tensor::eqsize(A, C))) msg("Incompatible dims", "Tensor::cross-entropy");

    C->tsem->lock();
    if (A->isCPU()) {
        cpu_cent(A, B, C);
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
