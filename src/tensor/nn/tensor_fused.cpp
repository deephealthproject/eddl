/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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


namespace tensorNN{

void conv2d_activation(string act, ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::conv2d_activation");

    if (D->I->isCPU()) {
        msg("NotImplementedError", "Tensor::conv2d_activation");
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
          msg("NotImplementedError", "Tensor::conv2d_activation");
      }
#endif
#ifdef cFPGA
    else {
        msg("NotImplementedError", "Tensor::Conv2D_Relu");
    }
#endif

}

// Conv2D + Maxpooling
void conv_stm(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::conv2d_activation");

    if (D->I->isCPU()) {
        msg("NotImplementedError", "Tensor::conv2d_activation");
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
          msg("NotImplementedError", "Tensor::conv2d_activation");
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv_stm(D);  
    }
#endif

}

// Conv2D + Maxpool
void conv_maxpool(ConvolDescriptor *D) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::conv2d_activation");

    if (D->I->isCPU()) {
        msg("NotImplementedError", "Tensor::conv2d_activation");
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
          msg("NotImplementedError", "Tensor::conv2d_activation");
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv_maxpool(D);  
    }
#endif

}

}
