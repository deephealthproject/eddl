/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


void gpu_range(Tensor *A, float start, float step) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    range<<<dimGrid,dimBlock>>>(A->ptr, start, step, A->size);
    check_cuda(cudaDeviceSynchronize(), "range");
}


void gpu_eye(Tensor *A, int offset) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    eye<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], A->shape[1], offset);
    check_cuda(cudaDeviceSynchronize(), "eye");
}
