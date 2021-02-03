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


//#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>


// GPU: Structs for Thrust ********************************************

struct is_positive {
    template <typename T>
    bool __device__ operator()(T v) {
        return v > 0;
    }
};


struct all_close {
    const float rtol;
    const float atol;
    const bool equal_nan;

    all_close(float rtol_, float atol_, bool equal_nan_) : rtol(rtol_), atol(atol_), equal_nan(equal_nan_)  { /* empty */ }

    __host__ __device__ bool operator()(float x, float y) const {
        return fabsf(x - y) <= (atol + rtol * fabsf(y));
    }
};


// CPU: Logic functions: Comparisons
void gpu_isfinite(Tensor *A, Tensor* B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_isfinite<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_isfinite");
}

void gpu_isinf(Tensor *A, Tensor* B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_isinf<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_isinf");
}

void gpu_isnan(Tensor *A, Tensor* B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_isnan<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_isnan");
}

void gpu_isneginf(Tensor *A, Tensor* B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_isneginf<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_isneginf");
}

void gpu_isposinf(Tensor *A, Tensor* B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_isposinf<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_isposinf");
}


// CPU: Logic functions: Comparisons
void gpu_logical_and(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_logical_and<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "logical_and");
}

void gpu_logical_or(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_logical_or<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "logical_or");
}

void gpu_logical_not(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_logical_not<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "logical_not");
}

void gpu_logical_xor(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_logical_xor<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "logical_xor");
}

// GPU: Logic functions: Truth value testing
bool gpu_all(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return thrust::transform_reduce(thrust::device, dev_ptr, dev_ptr+A->size, is_positive{}, true, thrust::logical_and<bool>{} );
}

bool gpu_any(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return thrust::transform_reduce(thrust::device, dev_ptr, dev_ptr+A->size, is_positive{}, false, thrust::logical_or<bool>{} );
}

bool gpu_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    int device=A->gpu_device;
    cudaSetDevice(device);


    thrust::device_ptr<float> A_dev_ptr = thrust::device_pointer_cast(A->ptr);
    thrust::device_ptr<float> B_dev_ptr = thrust::device_pointer_cast(B->ptr);

    thrust::device_vector<float> temp(A->size);
    thrust::transform(A_dev_ptr, A_dev_ptr+A->size, B_dev_ptr, temp.begin(), all_close(rtol, atol, equal_nan));
    return thrust::reduce(thrust::device, temp.begin(), temp.end(), true, thrust::logical_and<bool>{});
    // I think transform_reduce only supports one input vector
    // return thrust::transform_reduce(thrust::device, A_dev_ptr, A_dev_ptr+A->size, B_dev_ptr, all_close(rtol, atol, equal_nan), true, thrust::logical_and<bool>{} );
}

void gpu_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_isclose<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, rtol, atol, equal_nan, A->size);
    check_cuda(cudaDeviceSynchronize(), "isclose");
}

void gpu_greater(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_greater<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "greater");
}

void gpu_greater(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_greater<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "greater");
}

void gpu_greater_equal(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_greater_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "greater_equal");
}

void gpu_greater_equal(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_greater_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "greater_equal");
}

void gpu_less(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_less<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "less");
}

void gpu_less(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_less<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "less");
}

void gpu_less_equal(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_less_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "less_equal");
}

void gpu_less_equal(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_less_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "less_equal");
}

void gpu_equal(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "equal");
}

void gpu_equal(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "equal");
}

void gpu_not_equal(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_not_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "not_equal");
}

void gpu_not_equal(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_not_equal<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "not_equal");
}