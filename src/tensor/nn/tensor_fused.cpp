/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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

// Conv2D + Maxpooling + Add
void conv_stm_add(ConvolDescriptor *D,Tensor *Add) {
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
        fpga_conv_stm_add(D, Add);  
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

// Conv2D + ReLU + Maxpool
void conv_relu_maxpool(ConvolDescriptor *D) {
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
        fpga_conv_relu_maxpool(D);  
    }
#endif

}


// Conv2D + ReLU
void conv_relu(ConvolDescriptor *D) {
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
        fpga_conv_relu(D);  
    }
#endif

}

// Conv2D + LeakyReLU
void conv_leakyrelu(ConvolDescriptor *D, float alpha) {
    if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::conv2d_activation");

    if (D->I->isCPU()) {
        msg("NotImplementedError", "Tensor::conv_leakyrelu");
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
          msg("NotImplementedError", "Tensor::conv_leakyrelu");
      }
#endif
#ifdef cFPGA
    else {
        fpga_conv_leakyrelu(D, alpha);  
    }
#endif

}


}
