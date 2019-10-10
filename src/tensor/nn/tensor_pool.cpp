/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/gpu_tensor.h"
#include "../../hardware/gpu/gpu_hw.h"
#include "../../hardware/gpu/nn/gpu_nn.h"
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
