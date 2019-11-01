/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
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

void gpu_shift(Tensor *A, Tensor *B, vector<int> t_shift, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_shift; cudaMalloc((int**)&d_shift, 2*sizeof(int));
    cudaMemcpy(d_shift, t_shift.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    shift<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], d_shift, mode, constant);
    check_cuda(cudaDeviceSynchronize(),"shift");
}

void gpu_rotate(Tensor *A, Tensor *B, float angle, vector<int> axis, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

//    rotate<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"rotate");

}

void gpu_scale(Tensor *A, Tensor *B, int mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    scale<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], mode, constant);
    check_cuda(cudaDeviceSynchronize(),"scale");
}

void gpu_flip(Tensor *A, Tensor *B, int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    flip<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], axis);
    check_cuda(cudaDeviceSynchronize(),"flip");
}

void gpu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_coords_from; cudaMalloc((int**)&d_coords_from, coords_from.size()*sizeof(int));
    cudaMemcpy(d_coords_from, coords_from.data(), coords_from.size()*sizeof(int), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    int *d_coords_to; cudaMalloc((int**)&d_coords_to, coords_to.size()*sizeof(int));
    cudaMemcpy(d_coords_to, coords_to.data(), coords_to.size()*sizeof(int), cudaMemcpyHostToDevice);

    crop<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_coords_from, d_coords_to, constant);
    check_cuda(cudaDeviceSynchronize(),"crop");
}

void gpu_cutout(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    // Copy vector from host to device
    int *d_coords_from; cudaMalloc((int**)&d_coords_from, coords_from.size()*sizeof(int));
    cudaMemcpy(d_coords_from, coords_from.data(), coords_from.size()*sizeof(int), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    int *d_coords_to; cudaMalloc((int**)&d_coords_to, coords_to.size()*sizeof(int));
    cudaMemcpy(d_coords_to, coords_to.data(), coords_to.size()*sizeof(int), cudaMemcpyHostToDevice);

    cutout<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->shape[0], B->shape[1], B->shape[2], B->shape[3], d_coords_from, d_coords_to, constant);
    check_cuda(cudaDeviceSynchronize(),"cutout");
}
