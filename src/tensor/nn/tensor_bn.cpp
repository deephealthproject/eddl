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

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

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
#ifdef cFPGA
        else {
          fpga_permute_channels_last(A, B);
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
#ifdef cFPGA
        else {
          fpga_permute_channels_first(A, B);
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
#ifdef cFPGA
        else {
          fpga_permute_batch_last(A, B);
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
#ifdef cFPGA
        else {
          fpga_permute_batch_first(A, B);
        }
#endif
        PROFILING_FOOTER(permute_batch_last);
    }

}