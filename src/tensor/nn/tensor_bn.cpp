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
#include "eddl/hardware/cpu/cpu_tensor.h"
#include "eddl/profiling.h"

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

PROFILING_ENABLE_EXTERN(permute_channels_last);
PROFILING_ENABLE_EXTERN(permute_channels_first);
PROFILING_ENABLE_EXTERN(permute_batch_last);
PROFILING_ENABLE_EXTERN(permute_batch_first);

namespace tensorNN {


    void permute_channels_last(Tensor *A, Tensor *B) {

        PROFILING_HEADER(permute_channels_last);

        if (A->isCPU()) {
            cpu_permute_channels_last(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
            {
              gpu_permute_channels_last(A,B);
            }
#endif
        PROFILING_FOOTER(permute_channels_last);
        }

    void permute_channels_first(Tensor *A, Tensor *B) {

        PROFILING_HEADER(permute_channels_first);

        if (A->isCPU()) {
            cpu_permute_channels_first(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
            {
              gpu_permute_channels_first(A,B);
            }
#endif
        PROFILING_FOOTER(permute_channels_first);
    }


    void permute_batch_last(Tensor *A, Tensor *B) {

        PROFILING_HEADER(permute_batch_last);

        if (A->isCPU()) {
            cpu_permute_batch_last(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
            {
              gpu_permute_batch_last(A,B);
            }
#endif
        PROFILING_FOOTER(permute_batch_last);
    }

    void permute_batch_first(Tensor *A, Tensor *B) {

        PROFILING_HEADER(permute_batch_first);

        if (A->isCPU()) {
            cpu_permute_batch_first(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
            {
              gpu_permute_batch_first(A,B);
            }
#endif
        PROFILING_FOOTER(permute_batch_first);
    }

    void BatchNormForward(Tensor *input, Tensor *output, Tensor *opa,
                            Tensor *mean, Tensor *variance,
                            Tensor *bn_g, Tensor *bn_b,
                            Tensor *bn_mean, Tensor *bn_var,
                            bool trmode, float epsilon, float momentum)
    {
        if (input->isCPU()) {
            cpu_batchnorm_forward(input->shape[0], input->shape[1],
                input->ndim == 2 ? 1 : input->ndim == 3 ? input->shape[2] : input->shape[2] * input->shape[3],
                input->ptr, output->ptr, opa->ptr,
                mean->ptr, variance->ptr,
                bn_g != NULL ? bn_g->ptr : NULL,
                bn_b != NULL ? bn_b->ptr : NULL,
                bn_mean->ptr, bn_var->ptr, trmode, epsilon, momentum);
        } else if (input->isGPU()) {
#ifdef cGPU
            gpu_batchnorm_forward(input->gpu_device, input->shape[0], input->shape[1],
                input->ndim == 2 ? 1 : input->shape[2] * input->shape[3],
                input->ptr, output->ptr, opa->ptr,
                mean->ptr, variance->ptr,
                bn_g != NULL ? bn_g->ptr : NULL,
                bn_b != NULL ? bn_b->ptr : NULL,
                bn_mean->ptr, bn_var->ptr, trmode, epsilon, momentum);
#endif
        }
    }

    void BatchNormBackward(Tensor *delta, Tensor *opa, Tensor *pdelta,
                            Tensor *gbn_g, Tensor *gbn_b, Tensor *bn_g,
                            Tensor *bn_var,
                            Tensor *work1, Tensor *work2)
    {
        if (delta->isCPU()) {
            cpu_batchnorm_backward(delta->shape[0], delta->shape[1],
                delta->ndim == 2 ? 1 : delta->shape[2] * delta->shape[3],
                delta->ptr, opa->ptr, pdelta->ptr,
                gbn_g != NULL ? gbn_g->ptr : NULL,
                gbn_b != NULL ? gbn_b->ptr : NULL,
                bn_g != NULL ? bn_g->ptr : NULL,
                bn_var->ptr, work1->ptr, work2->ptr);
        } else if (delta->isGPU()) {
#ifdef cGPU
            gpu_batchnorm_backward(delta->gpu_device, delta->shape[0], delta->shape[1],
                delta->ndim == 2 ? 1 : delta->shape[2] * delta->shape[3],
                delta->ptr, opa->ptr, pdelta->ptr,
                gbn_g != NULL ? gbn_g->ptr : NULL,
                gbn_b != NULL ? gbn_b->ptr : NULL,
                bn_g != NULL ? bn_g->ptr : NULL,
                bn_var->ptr, work1->ptr, work2->ptr);
#endif
        }
    }

}
