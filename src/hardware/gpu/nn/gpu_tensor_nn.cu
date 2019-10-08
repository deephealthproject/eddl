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

    repeat_nn<<<dimGrid,dimBlock>>>(A->ptr, A->shape[2], A->shape[3], B->ptr, B->shape[2], B->shape[3], size.data(), size.size());
    check_cuda(cudaDeviceSynchronize(), "repeat_nn");
}

void gpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(D);

    d_repeat_nn<<<dimGrid,dimBlock>>>(D->ptr, D->shape[2], D->shape[3], A->ptr, A->shape[2], A->shape[3], size.data(), size.size());
    check_cuda(cudaDeviceSynchronize(), "d_repeat_nn");
}