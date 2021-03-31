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

PROFILING_ENABLE_EXTERN(Conv2D);
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


    if (D->I->isCPU()) {
#if 1
        cpu_conv2D(D); // Conv2D_grad depends on im2col stored in ptrI
        cpu_naive_conv2D(D, D->O->ptr);
#else
        int n = D->I->shape[0] * D->z*D->r*D->c;
        float *output = new float[n];
        cpu_naive_conv2D(D, output);
        cpu_conv2D(D);
        int pos = 0; float max = 0.0;
        for (int i = 0; i < n; i++) {
            float d = fabsf(output[i] - D->O->ptr[i]);
            if (fabs(D->O->ptr[i]) > 1e-7) d = d / fabsf(D->O->ptr[i]);
            if (d > max) { max = d; pos = i; }
        }
        printf("cpu_conv2D: %d %e (%e,%e)\n", pos, max, output[pos], D->O->ptr[pos]);
        delete output;
#endif
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
#if 1
        cpu_conv2D_grad(D);
        // cpu_naive_conv2D_grad(D, D->gK->ptr);
        // printf("cpu_conv2D_grad\n");
#else
        int n = D->kr * D->kc * D->kz * D->nk;
        float *output = new float[n];
        cpu_naive_conv2D_grad(D, output);
        cpu_conv2D_grad(D);
        int pos = 0; float max = 0.0;
        for (int i = 0; i < n; i++) {
            float d = fabsf(output[i] - D->gK->ptr[i]);
            if (fabs(D->gK->ptr[i]) > 1e-7) d = d / fabsf(D->gK->ptr[i]);
            if (d > max) { max = d; pos = i; }
        }
        printf("%d %e (%e,%e)\n", pos, max, output[pos], D->O->ptr[pos]);
        delete output;
#endif
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
#if 1
        // cpu_conv2D_back(D);
        cpu_naive_conv2D_back(D, D->ID->ptr);
#else
        int n = D->I->shape[0] * D->iz * D->ir * D->ic;
        float *output = new float[n];
        cpu_naive_conv2D_back(D, output);
        cpu_conv2D_back(D);
        int pos = 0; float max = 0.0;
        for (int i = 0; i < n; i++) {
            float d = fabsf(output[i] - D->ID->ptr[i]);
            if (fabs(D->ID->ptr[i]) > 1e-7) d = d / fabsf(D->ID->ptr[i]);
            if (d > max) { max = d; pos = i; }
        }
        printf("cpu_conv2D_back: %d %e (%e,%e)\n", pos, max, output[pos], D->ID->ptr[pos]);
        delete output;
#endif
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


    if (D->I->isCPU()) {
        cpu_conv3D(D);
    }
#ifdef cGPU
    else if (D->I->isGPU())
    {
        gpu_conv3D(D);
    }
#endif
#ifdef cFPGA
        else {
    fpga_conv3D(D);
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
#ifdef cFPGA
        else {
    fpga_conv3D_grad(D);
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
#ifdef cFPGA
        else {
    fpga_conv3D_back(D);
}
#endif

//    PROFILING_FOOTER(Conv3D_back);
}



    void Conv2DT(ConvolDescriptorT *D) {
        /////////////////////////////////////////////////////////////////////
        //// Conv2DT
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a ConvolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2DT");

//        PROFILING_HEADER(Conv2DT);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::Conv2DT");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_conv2DT(D);
        }
#endif
#ifdef cFPGA
            else {
            msg("NotImplementedError", "Tensor::Conv2DT");
    }
#endif


//        PROFILING_FOOTER(Conv2DT);
    }

    void Conv2DT_grad(ConvolDescriptorT *D) {
        /////////////////////////////////////////////////////////////////////
        //// Conv2DT Grad
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a ConvolDescriptor
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2DT");

//        PROFILING_HEADER(Conv2DT_grad);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::Conv2DT");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_conv2DT_grad(D);
        }
#endif
#ifdef cFPGA
            else {
            msg("NotImplementedError", "Tensor::Conv2DT");
    }
#endif


//        PROFILING_FOOTER(Conv2DT_grad);
    }

    void Conv2DT_back(ConvolDescriptorT *D) {
        /////////////////////////////////////////////////////////////////////
        //// Conv2DT Back
        //// Dimensions must be compatible
        //// A is input 4D Tensor, Batch x Channels x Rows x Cols
        //// D is a ConvolDescriptorT
        /////////////////////////////////////////////////////////////////////
        if ((D->I->ndim != 4)) msg("Tensors are not 4D", "Tensor::Conv2DT");

        PROFILING_HEADER(Conv2D_back);


        if (D->I->isCPU()) {
            msg("NotImplementedError", "Tensor::Conv2DT");
        }
#ifdef cGPU
        else if (D->I->isGPU())
        {
            gpu_conv2DT_back(D);
        }
#endif
#ifdef cFPGA
            else {
            msg("NotImplementedError", "Tensor::Conv2DT");
    }
#endif

//        PROFILING_FOOTER(Conv2DT_back);
    }


}
