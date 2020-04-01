/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "hardware/gpu/gpu_nn.h"
#include "hardware/gpu/gpu_nn_kernels.h"

#include "hardware/gpu/gpu_hw.h"
#include "hardware/gpu/gpu_tensor.h"
#include "hardware/gpu/gpu_kernels.h"

#include "tensor/tensor.h"
#include "descriptors/descriptors.h"


void gpu_cent(Tensor *A,Tensor *B,Tensor *C){

  int device=A->gpu_device;
  cudaSetDevice(device);
  setDims(A);

  cent<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_cent");
}
