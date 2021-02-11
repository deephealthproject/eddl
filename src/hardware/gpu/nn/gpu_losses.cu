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

//#include <thrust/transform.h>
/* #include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h> */

#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"

#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"


void gpu_cent(Tensor *A,Tensor *B,Tensor *C){

  int device=A->gpu_device;
  cudaSetDevice(device);
  setDims(A);

  cent<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_cent");
}

float gpu_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred){
    int device=y_true->gpu_device;
    cudaSetDevice(device);

    // Get Bacthes and Softmax dimension (classes)
    int n_batches = y_true->shape[0];
    int n_features = y_true->shape[1];

    // Calculate cuda blocks
    int blockSize = MAX_TPB;
    int numBlocks = (n_batches + blockSize - 1) / blockSize; // Same as: ceil(N/threads_block)

    float *sum_array;
    check_cuda(cudaMalloc((void**)&(sum_array), n_batches*sizeof(float)),"create temp array");
    check_cuda(cudaMemset(sum_array, 0, sizeof(float));
    check_cuda(cudaDeviceSynchronize(), "create");

    // Calculate derivative of Softmax
    gpu_categorical_cross_entropy<<<numBlocks, blockSize>>>(y_true->ptr, y_pred->ptr, sum_array, n_batches, n_features);
    check_cuda(cudaDeviceSynchronize(),"gpu_categorical_cross_entropy");

    // Reduce sum and compute mean
    // thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(sum_array);
    float sum_ce; // = thrust::reduce(dev_ptr, dev_ptr + n_batches);
    check_cuda(cudaMemcpy(&sum_ce, sum_array, sizeof(float), cudaMemcpyDeviceToHost);
    float mean_ce = -sum_ce;//(float)n_batches;  // Mean

    // Delete tmp array
    check_cuda(cudaFree(sum_array), "create temp array");

    return mean_ce;
}

void gpu_d_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta){
    int device=y_true->gpu_device;
    cudaSetDevice(device);

    setDims(y_true);

    gpu_d_categorical_cross_entropy<<<dimGrid, dimBlock>>>(y_true->ptr, y_pred->ptr, delta->ptr, y_true->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_d_categorical_cross_entropy");
}



float gpu_binary_cross_entropy(Tensor* y_true, Tensor* y_pred){
    int device=y_true->gpu_device;
    cudaSetDevice(device);

    // Get Bacthes and Softmax dimension (classes)
    int n_batches = y_true->shape[0];

    // Calculate cuda blocks
    int blockSize = MAX_TPB;
    int numBlocks = (n_batches + blockSize - 1) / blockSize; // Same as: ceil(N/threads_block)

    float *sum_array;
    check_cuda(cudaMalloc((void**)&(sum_array), n_batches*sizeof(float)),"create temp array");
    check_cuda(cudaMemset(sum_array, 0, sizeof(float));
    check_cuda(cudaDeviceSynchronize(), "create");

    // Calculate derivative of Softmax
    gpu_binary_cross_entropy<<<numBlocks, blockSize>>>(y_true->ptr, y_pred->ptr, sum_array, y_true->size);
    check_cuda(cudaDeviceSynchronize(),"gpu_binary_cross_entropy");

    // Reduce sum and compute mean
    // thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(sum_array);
    float sum_ce; // = thrust::reduce(dev_ptr, dev_ptr + n_batches);
    check_cuda(cudaMemcpy(&sum_ce, sum_array, sizeof(float), cudaMemcpyDeviceToHost);
    float mean_ce = -sum_ce;//(float)n_batches;  // Mean

    // Delete tmp array
    check_cuda(cudaFree(sum_array), "create temp array");

    return mean_ce;
}

void gpu_d_binary_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta){
    int device=y_true->gpu_device;
    cudaSetDevice(device);

    setDims(y_true);

    gpu_d_binary_cross_entropy<<<dimGrid, dimBlock>>>(y_true->ptr, y_pred->ptr, delta->ptr, y_true->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_d_binary_cross_entropy");
}
