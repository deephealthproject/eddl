/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "hardware/gpu/gpu_tensor.h"
#include "hardware/gpu/gpu_hw.h"
#include "hardware/gpu/nn/gpu_nn.h"
#endif



void permute_channels_last(Tensor *A,Tensor *B)
{
  if (A->isCPU()) {
        cpu_permute_channels_last(A,B);
  }
#ifdef cGPU
  else if (A->isGPU())
      {
        gpu_permute_channels_last(A,B);
      }
#endif
#ifdef cFPGA
  else {

    }
#endif
}

void permute_channels_first(Tensor *A,Tensor *B)
{
  if (A->isCPU()) {
        cpu_permute_channels_first(A,B);
  }
#ifdef cGPU
  else if (A->isGPU())
      {
        gpu_permute_channels_first(A,B);
      }
#endif
#ifdef cFPGA
  else {

    }
#endif
}


void permute_batch_last(Tensor *A,Tensor *B)
{
  if (A->isCPU()) {
        cpu_permute_batch_last(A,B);
  }
#ifdef cGPU
  else if (A->isGPU())
      {
        gpu_permute_batch_last(A,B);
      }
#endif
#ifdef cFPGA
  else {

    }
#endif
}

void permute_batch_first(Tensor *A,Tensor *B)
{
  if (A->isCPU()) {
        cpu_permute_batch_first(A,B);
  }
#ifdef cGPU
  else if (A->isGPU())
      {
        gpu_permute_batch_first(A,B);
      }
#endif
#ifdef cFPGA
  else {

    }
#endif
}
