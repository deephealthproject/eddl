#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/gpu_tensor.h"
#include "../../hardware/gpu/gpu_hw.h"
#include "../../hardware/gpu/nn/gpu_nn.h"
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
