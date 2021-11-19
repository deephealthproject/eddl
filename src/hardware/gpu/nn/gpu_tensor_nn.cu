/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <iostream>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>


#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"

#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"


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


void gpu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->stride[0]*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->stride[0]*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }


    setDims(B);  // B is the small
    gpu_select_nn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses, A->stride[0], B->stride[0]);
    check_cuda(cudaDeviceSynchronize(), "gpu_select_nn");
}

void gpu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, A->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), A->stride[0]*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, A->stride[0]*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }


    setDims(A);  // A is the small
    gpu_select_back_nn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, sd->gpu_addresses, A->stride[0], B->stride[0]);
    check_cuda(cudaDeviceSynchronize(), "gpu_select_back_nn");
}

void gpu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->stride[0]*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->stride[0]*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    gpu_set_select_nn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses, A->stride[0], B->stride[0]);
    check_cuda(cudaDeviceSynchronize(), "gpu_set_select_nn");
}

void gpu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->stride[0]*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->stride[0]*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    gpu_set_select_back_nn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses, A->stride[0], B->stride[0]);
    check_cuda(cudaDeviceSynchronize(), "gpu_set_select_back_nn");
}


void gpu_expand_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->stride[0]*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->stride[0]*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the larger one
    gpu_expand_nn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses, A->stride[0], B->stride[0]);
    check_cuda(cudaDeviceSynchronize(), "gpu_expand_nn");
}

void gpu_expand_back_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->stride[0]*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->stride[0]*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(A); // A is the larger one
    gpu_expand_back_nn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses, A->stride[0], B->stride[0]);
    check_cuda(cudaDeviceSynchronize(), "gpu_expand_back_nn");
}