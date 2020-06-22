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

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

namespace tensorNN {


    void permute_channels_last(Tensor *A, Tensor *B) {
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
    }

    void permute_channels_first(Tensor *A, Tensor *B) {
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
    }


    void permute_batch_last(Tensor *A, Tensor *B) {
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
    }

    void permute_batch_first(Tensor *A, Tensor *B) {
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
    }

}