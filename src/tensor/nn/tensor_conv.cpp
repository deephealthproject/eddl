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

PROFILING_ENABLE_EXTERN(Conv2D);
PROFILING_ENABLE_EXTERN(Conv2DReLU);
PROFILING_ENABLE_EXTERN(Conv2D_grad);
PROFILING_ENABLE_EXTERN(Conv2D_back);

PROFILING_ENABLE_EXTERN(Conv3D);
PROFILING_ENABLE_EXTERN(Conv3D_grad);
PROFILING_ENABLE_EXTERN(Conv3D_back);

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

    bool is_dilated = false;
    for (int i : D->dilation_rate)
        if (i > 1) {
            is_dilated = true;
            break;
        }

    if (D->I->isCPU()) {
        if (is_dilated)
            msg("Dilated convolutions are only supported using GPU with CUDNN." "Tensor::Conv2D");
        
        if (FixedPointQuant) msg("Fixed point quantization not available in CPU." "Tensor::Conv2D");
        else cpu_conv2D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
#ifndef cCUDNN
        if (is_dilated)
            msg("Dilated convolutions are only supported using GPU with CUDNN." "Tensor::Conv2D");
#endif
        //gpu_conv2D_old(D);
        gpu_conv2D(D);
      }
#endif

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


    if (D->I->isCPU()) {
        cpu_conv2D_grad(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         gpu_conv2D_grad(D);
      }
#endif
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


    if (D->I->isCPU()) {
        cpu_conv2D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
      {
         gpu_conv2D_back(D);
      }
#endif
    PROFILING_FOOTER(Conv2D_back);
}


void Conv3D(ConvolDescriptor3D *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv3D
    //// Dimensions must be compatible
    //// A is input 5D Tensor, batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
    //// D is a ConvolDescriptor3D
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 5)) msg("Tensors are not 5D", "Tensor::Conv3D");

//    PROFILING_HEADER(Conv3D);

    bool is_dilated = false;
    for (int i : D->dilation_rate)
        if (i > 1) {
            is_dilated = true;
            break;
        }

    if (D->I->isCPU()) {
        if (is_dilated)
            msg("Dilated convolutions are only supported using GPU with CUDNN." "Tensor::Conv3D");
        cpu_conv3D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
    {
#ifndef cCUDNN
        if (is_dilated)
            msg("Dilated convolutions are only supported using GPU with CUDNN." "Tensor::Conv3D");
#endif
        gpu_conv3D(D);
    }
#endif
//    PROFILING_FOOTER(Conv3D);
}

void Conv3D_grad(ConvolDescriptor3D *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv3D Grad
    //// Dimensions must be compatible
    //// A is input 5D Tensor, batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
    //// D is a ConvolDescriptor3D
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 5)) msg("Tensors are not 5D", "Tensor::Conv3D");

//    PROFILING_HEADER(Conv3D_grad);


    if (D->I->isCPU()) {
        cpu_conv3D_grad(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
    {
        gpu_conv3D_grad(D);
    }
#endif


//    PROFILING_FOOTER(Conv3D_grad);
}

void Conv3D_back(ConvolDescriptor3D *D) {
    /////////////////////////////////////////////////////////////////////
    //// Conv3D Back
    //// Dimensions must be compatible
    //// A is input 5D Tensor, batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
    //// D is a ConvolDescriptor3D
    /////////////////////////////////////////////////////////////////////
    if ((D->I->ndim != 5)) msg("Tensors are not 5D", "Tensor::Conv3D");

//    PROFILING_HEADER(Conv3D_back);


    if (D->I->isCPU()) {
        cpu_conv3D_back(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
    {
        gpu_conv3D_back(D);
    }
#endif

//    PROFILING_FOOTER(Conv3D_back);
}



    void ConvT2D(ConvolDescriptorT2D *D) {
        /////////////////////////////////////////////////////////////////////
        //// ConvT2D
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a ConvolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::ConvT2D");

//        PROFILING_HEADER(Conv2DT);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::ConvT2D");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_convT2D(D);
        }
#endif
//        PROFILING_FOOTER(Conv2DT);
    }

    void ConvT2D_grad(ConvolDescriptorT2D *D) {
        /////////////////////////////////////////////////////////////////////
        //// ConvT2D Grad
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a ConvolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::ConvT2D");

//        PROFILING_HEADER(Conv2DT_grad);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::ConvT2D");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_convT2D_grad(D);
        }
#endif
//        PROFILING_FOOTER(Conv2DT_grad);
    }

    void ConvT2D_back(ConvolDescriptorT2D *D) {
        /////////////////////////////////////////////////////////////////////
        //// ConvT2D Back
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a ConvolDescriptorT
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::ConvT2D");

        PROFILING_HEADER(Conv2D_back);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::ConvT2D");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_convT2D_back(D);
        }
#endif
//        PROFILING_FOOTER(Conv2DT_back);
    }



    void ConvT3D(ConvolDescriptorT3D *D) {
        /////////////////////////////////////////////////////////////////////
        //// ConvT3D
        //// Dimensions must be compatible
        //// A is input 5D Tensor, batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
        //// D is a ConvolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 5)) msg("Tensors are not 5D", "Tensor::ConvT3D");

//        PROFILING_HEADER(ConvT3D);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::ConvT3D");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_convT3D(D);
        }
#endif
//        PROFILING_FOOTER(ConvT3D);
    }

    void ConvT3D_grad(ConvolDescriptorT3D *D) {
        /////////////////////////////////////////////////////////////////////
        //// ConvT3D Grad
        //// Dimensions must be compatible
        //// A is input 5D Tensor, batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
        //// D is a ConvolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 5)) msg("Tensors are not 5D", "Tensor::ConvT3D_grad");

//        PROFILING_HEADER(ConvT3D_grad);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::ConvT3D_grad");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_convT3D_grad(D);
        }
#endif
//        PROFILING_FOOTER(ConvT3D_grad);
    }

    void ConvT3D_back(ConvolDescriptorT3D *D) {
        /////////////////////////////////////////////////////////////////////
        //// ConvT3D Back
        //// Dimensions must be compatible
        //// A is input 5D Tensor, batch_shape + (channels, conv_dim1, conv_dim2, conv_dim3)
        //// D is a ConvolDescriptorT
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 5)) msg("Tensors are not 5D", "Tensor::ConvT3D_back");

//        PROFILING_HEADER(ConvT3D_back);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::ConvT3D_back");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_convT3D_back(D);
        }
#endif
//        PROFILING_FOOTER(ConvT3D_back);
    }

}
