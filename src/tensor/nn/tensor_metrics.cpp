/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
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

PROFILING_ENABLE_EXTERN(accuracy);
PROFILING_ENABLE_EXTERN(bin_accuracy);

namespace tensorNN {


    int accuracy(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::accuracy");
        if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x probs)", "Tensor::Accuracy");

        PROFILING_HEADER(accuracy);

        int acc = 0;



        if (A->isCPU()) {
            acc = cpu_accuracy(A, B);
        }
#ifdef cGPU
        else if (A->isGPU()) {
            gpu_accuracy(A, B, &acc);
        }
#endif
        PROFILING_FOOTER(accuracy);

        return acc;

    }

    int bin_accuracy(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::accuracy");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::accuracy");
        if (A->ndim != 2) msg("Accuracy only over 2D Tensor (batch x prob)", "Tensor::Bin_Accuracy");

        if (A->shape[1] != 1)
            msg("Accuracy only over 2D Tensor (batch x prob) within shape:{batchx1}", "Tensor::Bin_Accuracy");

        PROFILING_HEADER(bin_accuracy);

        int acc = 0;



        if (A->isCPU()) {
            acc = cpu_bin_accuracy(A, B);
        }
#ifdef cGPU
        else if (A->isGPU()) {
            gpu_bin_accuracy(A, B, &acc);
        }
#endif
        PROFILING_FOOTER(bin_accuracy);
        
        return acc;

    }

}
