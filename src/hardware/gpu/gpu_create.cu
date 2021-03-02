/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"
#include "eddl/hardware/gpu/gpu_hw.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"


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


void gpu_diag(Tensor *A, Tensor *B, int k){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_diag<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], k);
    check_cuda(cudaDeviceSynchronize(), "diag");

}
