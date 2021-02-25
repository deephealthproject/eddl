/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/profiling.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

PROFILING_ENABLE_EXTERN(Conv2D);
PROFILING_ENABLE_EXTERN(Conv2DReLU);
PROFILING_ENABLE_EXTERN(Conv2D_grad);
PROFILING_ENABLE_EXTERN(Conv2D_back);

namespace tensorNN{

void Conv2D(ConvolDescriptor *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv2D
    //// Dimensions must be compatible
    //// A is input 4D Tensor, Batch x Channels x Rows x Cols
    //// D is a ConvolDescriptor
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    PROFILING_HEADER(Conv2D);

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         //gpu_conv2D_old(D);
         gpu_conv2D(D);
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv2D(D);
    }
#endif
    D->O->tsem->unlock();

    PROFILING_FOOTER(Conv2D);
}

void Conv2D_grad(ConvolDescriptor *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv2D Grad
    //// Dimensions must be compatible
    //// A is input 4D Tensor, Batch x Channels x Rows x Cols
    //// D is a ConvolDescriptor
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    PROFILING_HEADER(Conv2D_grad);

    D->gK->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D_grad(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         gpu_conv2D_grad(D);
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv2D_grad(D);
    }
#endif
    D->gK->tsem->unlock();

    PROFILING_FOOTER(Conv2D_grad);
}

void Conv2D_back(ConvolDescriptor *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv2D Back
    //// Dimensions must be compatible
    //// A is input 4D Tensor, Batch x Channels x Rows x Cols
    //// D is a ConvolDescriptor
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    PROFILING_HEADER(Conv2D_back);

    D->ID->tsem->lock();
    if (D->I->isCPU()) {
        cpu_conv2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         gpu_conv2D_back(D);
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv2D_back(D);
    }
#endif
    D->ID->tsem->unlock();

    PROFILING_FOOTER(Conv2D_back);
}

void Conv2DReLU(ConvolDescriptor *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv2DReLU
    //// Dimensions must be compatible
    //// A is input 4D Tensor, Batch x Channels x Rows x Cols
    //// D is a ConvolDescriptor
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2D");

    PROFILING_HEADER(Conv2DReLU);

    D->O->tsem->lock();
    if (D->I->isCPU()) {
        printf("Error, Conv2DReLU not supported in CPU\n");
        exit(1);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
          printf("Error, Conv2DReLU not supported in GPU\n");
          exit(1);
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv2DReLU(D);
    }
#endif
    D->O->tsem->unlock();

    PROFILING_FOOTER(Conv2DReLU);
}

}
