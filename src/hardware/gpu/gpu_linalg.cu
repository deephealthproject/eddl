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

#include <thrust/device_ptr.h>
//#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"
#include "eddl/hardware/gpu/gpu_hw.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"



// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct sqr
{
    __host__ __device__
    T operator()(const T& x) const {
        return x * x;
    }
};

float gpu_norm(Tensor *A, string ord){
    int device=A->gpu_device;
    cudaSetDevice(device);

    if (ord=="fro"){
        thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);

        // setup arguments
        sqr<float>        unary_op;
        thrust::plus<float> binary_op;
        float init = 0;

        float abs_sum_sqr = thrust::transform_reduce(dev_ptr, dev_ptr + A->size, unary_op, init, binary_op);
        float norm = std::sqrt(abs_sum_sqr);

        return norm;
    }else{
        msg("Not yet implemented", "gpu_norm");
    }

    return 0.0f;
}


void gpu_norm(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, string ord){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true); // Walk through the source tensor
    setDims(B);

    if (ord=="fro") {
        gpu_norm_fro<<<dimGrid, dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    }else{
        msg("Not yet implemented", "gpu_norm");
    }

    check_cuda(cudaDeviceSynchronize(),"reduce_norm");
}
