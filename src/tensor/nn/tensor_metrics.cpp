/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

namespace tensorNN {


    int accuracy(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::accuracy");
        if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x probs)", "Tensor::Accuracy");

        int acc = 0;

        B->tsem->lock();

        if (A->isCPU()) {
            acc = cpu_accuracy(A, B);
        }
#ifdef cGPU
        else if (A->isGPU()) {
            gpu_accuracy(A, B, &acc);
        }
#endif
#ifdef cFPGA
        else {

        }
#endif
        B->tsem->unlock();
        return acc;

    }

    int bin_accuracy(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::accuracy");
        if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x prob)", "Tensor::Bin_Accuracy");

        if (A->shape[1] != 1)
            msg("Accuracy only over 2D Tensor (batch x prob) within shape:{batchx1}", "Tensor::Bin_Accuracy");


        int acc = 0;

        B->tsem->lock();

        if (A->isCPU()) {
            acc = cpu_bin_accuracy(A, B);
        }
#ifdef cGPU
        else if (A->isGPU()) {
            gpu_bin_accuracy(A, B, &acc);
        }
#endif
#ifdef cFPGA
        else {

        }
#endif
        B->tsem->unlock();
        return acc;

    }

}