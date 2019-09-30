#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/tensor_cuda.h"
#include "../../hardware/gpu/tensor_cuda_op.h"
#endif



void MPool2D(PoolDescriptor *D) {
    /////////////////////////////////////////////////////////////////////
    //// MPool2D
    //// Dimensions must be compatible
    //// A is input 4D Tensor, Batch x Channels x Rows x Cols
    //// D is a ConvolDescriptor
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D");

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        cpu_mpool2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
        gpu_mpool2D(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->O->tsem->unlock();
}

void MPool2D_back(PoolDescriptor *D) {
    /////////////////////////////////////////////////////////////////////
    //// MPool2D
    //// Dimensions must be compatible
    //// A is input 4D Tensor, Batch x Channels x Rows x Cols
    //// D is a ConvolDescriptor
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::MPool2D_back");

    D->ID->tsem->lock();
    if (D->I->isCPU()) {

        cpu_mpool2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
        gpu_mpool2D_back(D);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    D->ID->tsem->unlock();
}
