/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
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

PROFILING_ENABLE_EXTERN(cent);

#include <cmath>

namespace tensorNN {


// Cross-Entropy: C=-(A*log(B)+(1-A)*log_(1-B))
    void cent(Tensor *A, Tensor *B, Tensor *C) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::cross-entropy");
        if ((!Tensor::sameShape(A, B)) || (!Tensor::sameShape(A, C))) msg("Incompatible dims", "Tensor::cross-entropy");

        PROFILING_HEADER(cent);


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
        else if (A->isFPGA())
      {
         fpga_cent(A,B,C);
      }
#endif


        PROFILING_FOOTER(cent);
    }


    float categorical_cross_entropy(Tensor* y_true, Tensor* y_pred){
        if (!Tensor::sameDevice(y_true, y_pred)) {
            msg("Tensors in different devices", "TensorNN::categorical_cross_entropy");
        }
        if (!Tensor::sameShape(y_true, y_pred)) {
            msg("Incompatible dims", "TensorNN::categorical_cross_entropy");
        }

        if (y_true->isCPU()) {
            return cpu_categorical_cross_entropy(y_true, y_pred);
        }
#ifdef cGPU
        else if (y_true->isGPU())
        {
            return gpu_categorical_cross_entropy(y_true, y_pred);
        }
#endif
#ifdef cFPGA
        else {
        return fpga_categorical_cross_entropy(y_true, y_pred);
    }
#endif
        return std::nanf("");
    }

    void d_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta){
        if (!Tensor::sameDevice(y_true, y_pred) || !Tensor::sameDevice(y_true, delta)) {
            msg("Tensors in different devices", "TensorNN::d_categorical_cross_entropy");
        }
        if (!Tensor::sameShape(y_true, y_pred) || !Tensor::sameShape(y_true, delta)) {
            msg("Incompatible dims", "TensorNN::d_categorical_cross_entropy");
        }

        if (y_true->isCPU()) {
            cpu_d_categorical_cross_entropy(y_true, y_pred, delta);
        }
#ifdef cGPU
        else if (y_true->isGPU())
        {
            gpu_d_categorical_cross_entropy(y_true, y_pred, delta);
        }
#endif
#ifdef cFPGA
        else {
        fpga_d_categorical_cross_entropy(y_true, y_pred, delta);
    }
#endif
    }

    float binary_cross_entropy(Tensor* y_true, Tensor* y_pred){
        if (!Tensor::sameDevice(y_true, y_pred)) {
            msg("Tensors in different devices", "TensorNN::binary_cross_entropy");
        }
        if (!Tensor::sameShape(y_true, y_pred)) {
            msg("Incompatible dims", "TensorNN::binary_cross_entropy");
        }
        if (y_true->isCPU()) {
            return cpu_binary_cross_entropy(y_true, y_pred);
        }
#ifdef cGPU
        else if (y_true->isGPU())
        {
            return gpu_binary_cross_entropy(y_true, y_pred);
        }
#endif
#ifdef cFPGA
        else {
        return fpga_full_cross_entropy(y_true, y_pred);
    }
#endif
        return std::nanf("");
    }

    void d_binary_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta){
        if (!Tensor::sameDevice(y_true, y_pred) || !Tensor::sameDevice(y_true, delta)) {
            msg("Tensors in different devices", "TensorNN::d_binary_cross_entropy");
        }
        if (!Tensor::sameShape(y_true, y_pred) || !Tensor::sameShape(y_true, delta)) {
            msg("Incompatible dims", "TensorNN::d_binary_cross_entropy");
        }

        if (y_true->isCPU()) {
            cpu_d_binary_cross_entropy(y_true, y_pred, delta);
        }
#ifdef cGPU
        else if (y_true->isGPU())
        {
            gpu_d_binary_cross_entropy(y_true, y_pred, delta);
        }
#endif
#ifdef cFPGA
        else {
        fpga_d_binary_cross_entropy(y_true, y_pred, delta);
    }
#endif
    }


}  // namespace
