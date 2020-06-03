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

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

namespace tensorNN {


    void repeat_nn(Tensor *A, Tensor *B, vector<int> size) {
        // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)

        if ((A->device != B->device)) msg("Tensors in different devices", "Tensor::Repeat_NN");
        if (A->ndim != B->ndim) msg("Incompatible dims", "Tensor::Repeat");

        // Check size
        for (int i = 2; i < A->ndim; i++) {
            if (A->shape[i] * size[i - 2] != B->shape[i]) {
                msg("Incompatible dimensions (size)", "Tensor::Repeat_NN");
            }
        }

        if (A->isCPU() && B->isCPU()) {
            cpu_repeat_nn(A, B, size);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU()) {
            gpu_repeat_nn(A, B, size);
        }
#endif
#ifdef cFPGA
        else {

        }
#endif
    }

    void d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
        // TODO: Should be for N dimensions, not 2 (...and generic, not just NN)
        if ((D->device != A->device)) msg("Tensors in different devices", "Tensor::D_Repeat_NN");

        if (D->isCPU() && A->isCPU()) {
            cpu_d_repeat_nn(D, A, size);
        }
#ifdef cGPU
        else if (D->isGPU() && A->isGPU()) {
            gpu_d_repeat_nn(D, A, size);
        }
#endif
#ifdef cFPGA
        else {

        }
#endif
    }


    void select(Tensor *A, Tensor* B, SelDescriptor *sd){
        if (A->isCPU() && B->isCPU()) {
            cpu_select_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            msg("Not yet implemented", "Tensor::select");
            //gpu_select_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
        else {

    }
#endif

    }

    void select_back(Tensor *A, Tensor* B, SelDescriptor *sd){
        if (A->isCPU() && B->isCPU()) {
            cpu_select_back_nn(A, B, sd);
        }
#ifdef cGPU
        else if (A->isGPU() && B->isGPU())
        {
            msg("Not yet implemented", "Tensor::select_back");
//            gpu_select_back_nn(A, B, sd);
        }
#endif
#ifdef cFPGA
        else {

    }
#endif

    }

}