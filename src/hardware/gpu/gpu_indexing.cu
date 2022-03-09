/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
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


// GPU: Logic functions: Comparisons
void gpu_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_where<<<dimGrid,dimBlock>>>(condition->ptr, A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_where");
}


// GPU: Logic functions: Comparisons
void gpu_where_back(Tensor *condition, Tensor *PD_A, Tensor *PD_B, Tensor *D){
    int device=PD_A->gpu_device;
    cudaSetDevice(device);

    setDims(PD_A);

    gpu_where_back<<<dimGrid,dimBlock>>>(condition->ptr, PD_A->ptr, PD_B->ptr, D->ptr, PD_A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_where_back");
}