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

Tensor* gpu_shift(Tensor *A, vector<int> t_shift, string mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    Tensor *B = Tensor::full(A->getShape(), constant, A->device);
    setDims(B);

    // Copy vector from host to device
    int *d_shift; cudaMalloc((int**)&d_shift, 2*sizeof(int));
    cudaMemcpy(d_shift, t_shift.data(), 2*sizeof(int), cudaMemcpyHostToDevice);

    int d_mode = 0;
    shift<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], d_shift, d_mode, constant);
    check_cuda(cudaDeviceSynchronize(),"shift");

    return B;
}

Tensor* gpu_rotate(Tensor *A, float angle, vector<int> axis, bool reshape, string mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

//    rotate<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"rotate");

    return A;
}

Tensor* gpu_scale(Tensor *A, vector<int> new_shape, bool reshape, string mode, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    Tensor *B;
    int offsets[2] = {0, 0};
    int *d_offsets; cudaMalloc((int**)&d_offsets, 2*sizeof(int));

    // Resize keeping the original size (e.g.: if zoom-out, add zeros, else "crop")
    if(reshape) { B = Tensor::full(new_shape, constant, A->device);
    } else {
        B = Tensor::full(A->getShape(), constant, A->device);

        // Compute offset to center the inner matrix (zoom-out)
        offsets[0] = A->shape[0]/2.0f - new_shape[2]/2.0f;
        offsets[1] = A->shape[1]/2.0f - new_shape[3]/2.0f;
    }
    cudaMemcpy(d_offsets, offsets, 2*sizeof(int), cudaMemcpyHostToDevice);   
    setDims(B);

    int d_mode = 0;
    scale<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], B->shape[2], B->shape[3], d_offsets, d_mode, constant);
    check_cuda(cudaDeviceSynchronize(),"scale");
    return B;
}

Tensor* gpu_flip(Tensor *A, int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    Tensor* B = new Tensor(A->getShape(), A->device);
    setDims(B);

    flip<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], axis);
    check_cuda(cudaDeviceSynchronize(),"flip");
    return B;
}

Tensor* gpu_crop(Tensor *A, vector<int> coords_from, vector<int> coords_to, bool reshape, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    Tensor *B;
    vector<int> new_shape;

    // If "True", return a smaller tensor. Else, fill the non-cropped region
    if(reshape) {
        for(int i=0; i<A->ndim; i++){
            new_shape.push_back(coords_to[i] - coords_from[i] + 1);
        }
    } else { new_shape = A->shape; }

    B = Tensor::full(new_shape, constant);
    setDims(B);

    // Copy vector from host to device
    int *d_coords_from; cudaMalloc((int**)&d_coords_from, coords_from.size()*sizeof(int));
    cudaMemcpy(d_coords_from, coords_from.data(), coords_from.size()*sizeof(int), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    int *d_coords_to; cudaMalloc((int**)&d_coords_to, coords_to.size()*sizeof(int));
    cudaMemcpy(d_coords_to, coords_to.data(), coords_to.size()*sizeof(int), cudaMemcpyHostToDevice);

    //crop<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->shape[0], A->shape[1], A->shape[2], A->shape[3], d_coords_from, coords_from.size(), d_coords_to, coords_to.size());
    check_cuda(cudaDeviceSynchronize(),"crop");
    return B;
}

Tensor* gpu_cutout(Tensor *A, vector<int> coords_from, vector<int> coords_to, float constant){
    int device=A->gpu_device;
    cudaSetDevice(device);

    Tensor* B = A->clone();
    setDims(B);

    // Copy vector from host to device
    int *d_coords_from; cudaMalloc((int**)&d_coords_from, coords_from.size()*sizeof(int));
    cudaMemcpy(d_coords_from, coords_from.data(), coords_from.size()*sizeof(int), cudaMemcpyHostToDevice);

    // Copy vector from host to device
    int *d_coords_to; cudaMalloc((int**)&d_coords_to, coords_to.size()*sizeof(int));
    cudaMemcpy(d_coords_to, coords_to.data(), coords_to.size()*sizeof(int), cudaMemcpyHostToDevice);


    cutout<<<dimGrid,dimBlock>>>(B->ptr, B->shape[0], B->shape[1], B->shape[2], B->shape[3], d_coords_from, coords_from.size(), d_coords_to, coords_to.size(), constant);
    check_cuda(cudaDeviceSynchronize(),"cutout");
    return B;
}
