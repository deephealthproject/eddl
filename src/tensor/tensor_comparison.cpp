#include "tensor.h"
#include "../hardware/cpu/cpu_hw.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;

int Tensor::eqsize(Tensor *A, Tensor *B) {
    if (A->ndim != B->ndim) return 0;

    for (int i = 0; i < A->ndim; i++)
        if (A->shape[i] != B->shape[i]) return 0;

    return 1;
}

int Tensor::equal(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::equal");

    if (!eqsize(A,B)) return 0;

    if (A->isCPU()) {
        cpu_equal(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
          {
            msg("Equal only for CPU Tensors", "Tensor::equal");
          }
#endif
#ifdef cFPGA
    else {
          msg("Equal only for CPU Tensors", "Tensor::equal");
        }
#endif

    return 1;
}
