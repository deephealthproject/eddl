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

Tensor* gpu_shift(Tensor *A, vector<int> shift, string mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    Tensor *B = Tensor::full(A->getShape(), constant, A->device);
    setDims(B);

    // Copy vector from host to device
    int *d_shift; cudaMalloc((int**)&d_shift, 2*sizeof(int));
    cudaMemcpy(d_shift, shift.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    int mode = 0;
    shift<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[2], d_shift, mode, constant);
    check_cuda(cudaDeviceSynchronize(),"shift");
}

Tensor* gpu_rotate(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

//    rotate<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"rotate");
}

Tensor* gpu_scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

//    scale<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"scale");
}

Tensor* gpu_flip(Tensor *A, int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

//    flip<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"flip");
}

Tensor* gpu_crop(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

//    crop<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"crop");
}

Tensor* gpu_cutout(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

//    cutout<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"cutout");
}