/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>


#include "gpu_nn.h"
#include "gpu_nn_kernels.h"

#include "../gpu_hw.h"
#include "../gpu_tensor.h"
#include "../gpu_kernels.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


void gpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_size; cudaMalloc((int**)&d_size, 2*sizeof(int));
    cudaMemcpy(d_size, size.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    repeat_nn_k<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->ptr, B->shape[2], B->shape[3], d_size);

    cudaFree(d_size);
    check_cuda(cudaDeviceSynchronize(), "repeat_nn_k");
}

void gpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(D);

    // Copy vector from host to device
    int *d_size; cudaMalloc((int**)&d_size, 2*sizeof(int));
    cudaMemcpy(d_size, size.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    d_repeat_nn_k<<<dimGrid,dimBlock>>>(D->ptr, D->shape[0], D->shape[1], D->shape[2], D->shape[3], A->ptr, A->shape[2], A->shape[3], d_size);

    cudaFree(d_size);
    check_cuda(cudaDeviceSynchronize(), "d_repeat_nn_k");
}
