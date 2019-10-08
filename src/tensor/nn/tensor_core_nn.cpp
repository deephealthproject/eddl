#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/gpu_tensor.h"
#include "../../hardware/gpu/gpu_hw.h"
#include "../../hardware/gpu/nn/gpu_nn.h"
#endif


void repeat_nn(Tensor *A, Tensor *B, vector<int> size) {
    // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)

    if ((A->device != B->device)) msg("Tensors in different devices", "Tensor::Repeat_NN");
    if (A->ndim != B->ndim) msg("Incompatible dims", "Tensor::Repeat");

    // Check size
    for(int i=2; i<A->ndim; i++){
        if(A->shape[i]*size[i-2]!=B->shape[i]){
            msg("Incompatible dimensions (size)", "Tensor::Repeat_NN");
        }
    }

    if (A->isCPU() && B->isCPU()) {
        cpu_repeat_nn(A, B, size);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU()) {
      {
        gpu_repeat_nn(A, B, size);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}

void d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
    // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)
    if ((A->device != D->device)) msg("Tensors in different devices", "Tensor::D_Repeat_NN");

    if (A->isCPU() && D->isCPU()) {
        cpu_d_repeat_nn(D, A, size);
    }
#ifdef cGPU
    else if (A->isGPU() && B->isGPU()) {
      {
        gpu_repeat_nn(A, B, size);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
}