/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>

#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"
#include "eddl/hardware/gpu/gpu_hw.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

// GPU: Structs for Thrust ********************************************

template<typename T>
struct absolute_value : public unary_function<T,T>
{
    __host__ __device__ T operator()(const T &x) const
    {
        return x < T(0) ? -x : x;
    }
};


/*
 * @struct varianceshifteop
 * @brief a unary function that shifts input data
 * by their mean and computes the squares of them
 */
struct variance_shift_sum : std::unary_function<float, float>{
    const float mean;

    variance_shift_sum(float m) : mean(m) { /* empty */ }

    __host__ __device__ float operator()(float x) const {
        float tmp = x - mean;
        return tmp*tmp;
    }
};

template <class T>
struct bigger_tuple {
    __device__ __host__
    tuple<T,int> operator()(const tuple<T,int> &a, const tuple<T,int> &b)
    {
        if (a > b) return a;
        else return b;
    }

};

template <class T>
int max_index(thrust::device_vector<T>& vec) {

    // create implicit index sequence [0, 1, 2, ... )
    thrust::counting_iterator<int> begin(0); thrust::counting_iterator<int> end(vec.size());
    tuple<T,int> init(vec[0],0);
    tuple<T,int> smallest;

    smallest = reduce(make_zip_iterator(make_tuple(vec.begin(), begin)), make_zip_iterator(make_tuple(vec.end(), end)), init, bigger_tuple<T>());
    return get<1>(smallest);
}

// GPU: Math (in-place) ********************************************
void gpu_abs(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_abs<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "abs");
}

void gpu_acos(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_acos<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "acos");
}

void gpu_add(Tensor *A, Tensor *B, float v) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_add<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "add");
}

void gpu_asin(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_asin<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "asin");
}

void gpu_atan(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_atan<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "atan");
}

void gpu_ceil(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_ceil<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "ceil");
}

void gpu_clamp(Tensor *A, Tensor *B, float min, float max){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_clamp<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, min, max);
    check_cuda(cudaDeviceSynchronize(), "clamp");
}

void gpu_cos(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_cos<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "cos");
}

void gpu_cosh(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_cosh<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "cosh");
}

void gpu_exp(Tensor *A, Tensor *B){

    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_exp<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"exp");

}

void gpu_floor(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_floor<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "floor");
}


void gpu_log(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_log<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "log");

}


void gpu_log2(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_log2<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"log2");
}


void gpu_log10(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_log10<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"log10");
}


void gpu_logn(Tensor *A, Tensor *B, float n){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_logn<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, n);
    check_cuda(cudaDeviceSynchronize(), "logn");
};

void gpu_mod(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_mod<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "mod");
}

void gpu_inv(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_inv<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(),"inv");
}

void gpu_mult(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_mult<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(),"mult");

}

void gpu_normalize(Tensor *A, Tensor *B, float min, float max){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    float min_ori = gpu_min(A);
    float max_ori = gpu_max(A);

    gpu_normalize<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, min_ori, max_ori, min, max);
    check_cuda(cudaDeviceSynchronize(), "normalize");
}

void gpu_pow(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_pow<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "pow");
}

void gpu_powb(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_powb<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "powb");
}

void gpu_remainder(Tensor *A, Tensor *B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_remainder<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "remainder");
}

void gpu_round(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_round<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "round");
}

void gpu_rsqrt(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_rsqrt<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "rsqrt");
}

void gpu_sigmoid(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sigmoid<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sigmoid");
}

void gpu_sign(Tensor *A, Tensor *B, float zero_sign){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sign<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, zero_sign);
    check_cuda(cudaDeviceSynchronize(), "sign");
}

void gpu_sin(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sin<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sin");
}

void gpu_sinh(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sinh<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sinh");
}


void gpu_sqr(Tensor *A, Tensor *B){

    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sqr<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"sqr");

}

void gpu_sqrt(Tensor *A, Tensor *B){

    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sqrt<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(),"sqrt");
}

void gpu_tan(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_tan<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "tan");
}

void gpu_tanh(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_tanh<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "tanh");
}

void gpu_trunc(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_trunc<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "trunc");
}

// CPU: Math (static) ********************************************


void gpu_add(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_add<<<dimGrid,dimBlock>>>(scA,A->ptr,scB,B->ptr,C->ptr,incC,A->size);
    check_cuda(cudaDeviceSynchronize(),"addc");
}


void gpu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C,int incC){
    int device=A->gpu_device;
    cudaSetDevice(device);

    float alfa=1.0;
    float beta=(float)incC;

    cublasOperation_t trA = CUBLAS_OP_N;
    cublasOperation_t trB = CUBLAS_OP_N;

    int ldA=A->shape[1];
    int ldB=B->shape[1];
    int ldC=B->shape[1];
    int m=B->shape[1];
    int n=A->shape[0];
    int k=B->shape[0];


    if (tA)
    {
        trA = CUBLAS_OP_T;
        n=A->shape[1];
    }
    if (tB)
    {
        trB = CUBLAS_OP_T;
        m=B->shape[0];
        k=B->shape[1];
        ldC=B->shape[0];
    }

    check_cublas(cublasSgemm(hcublas[device],trB,trA,m,n,k,&alfa,B->ptr,ldB,A->ptr,ldA,&beta,C->ptr,ldC),"mult2D");

}


void gpu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_el_div<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->size);

    check_cuda(cudaDeviceSynchronize(),"gpu_el_div");
}


void gpu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_el_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->size);

    check_cuda(cudaDeviceSynchronize(),"gpu_el_mult");
}


void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);


    gpu_sum2D_rowwise<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1]);

    check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}


void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_sum2D_colwise<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1]);

    check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}

void gpu_maximum(Tensor* A, Tensor* B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_maximum<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "maximum");
}

void gpu_maximum(Tensor* A, Tensor* B, Tensor* C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_maximum<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "maximum");
}

void gpu_minimum(Tensor* A, Tensor* B, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_minimum<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, v, A->size);
    check_cuda(cudaDeviceSynchronize(), "minimum");
}

void gpu_minimum(Tensor* A, Tensor* B, Tensor* C){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    gpu_minimum<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, C->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "minimum");
}


// GPU: Should be reductions ***************************
float gpu_max(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return *thrust::max_element(dev_ptr, dev_ptr + A->size);
}

void gpu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_max<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, false);
    check_cuda(cudaDeviceSynchronize(),"reduce_max");
}

int gpu_argmax(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    thrust::device_ptr<float> max_ptr = thrust::max_element(dev_ptr, dev_ptr+A->size);

    //float max = *max_ptr;
    int argmax = (max_ptr - dev_ptr);
    return argmax;
}

void gpu_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_max<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, true);
    check_cuda(cudaDeviceSynchronize(),"reduce_argmax");
}


float gpu_min(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return *thrust::min_element(dev_ptr, dev_ptr + A->size);
}


void gpu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_min<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, false);
    check_cuda(cudaDeviceSynchronize(),"reduce_min");
}

int gpu_argmin(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    thrust::device_ptr<float> min_ptr = thrust::min_element(dev_ptr, dev_ptr+A->size);

//    float min = *min_ptr;
    int argmin = (min_ptr - dev_ptr);
    return argmin;
}

void gpu_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_min<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, true);
    check_cuda(cudaDeviceSynchronize(),"reduce_argmin");
}

float gpu_sum(Tensor *A){
    int device=A->gpu_device;

    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return thrust::reduce(dev_ptr, dev_ptr + A->size);
}

void gpu_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_sum<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    check_cuda(cudaDeviceSynchronize(),"reduce_sum");
}


float gpu_sum_abs(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return thrust::transform_reduce(dev_ptr, dev_ptr + A->size, absolute_value<float>(), 0.0f, thrust::plus<float>());
}

void gpu_sum_abs(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_sum_abs<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    check_cuda(cudaDeviceSynchronize(),"reduce_sum_abs");
}

float gpu_prod(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    float prod=thrust::reduce(dev_ptr, dev_ptr + A->size, 1.0f, thrust::multiplies<float>());

    return prod;
}

void gpu_prod(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_prod<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    check_cuda(cudaDeviceSynchronize(),"reduce_prod");
}


float gpu_mean(Tensor *A){
    int device=A->gpu_device;

    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return thrust::reduce(dev_ptr, dev_ptr + A->size)/(float)A->size;
}

void gpu_mean(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);


    setDims(B);  // Walk through reduced tensor
    gpu_mean<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    check_cuda(cudaDeviceSynchronize(),"reduce_gpu_mean");
}


float gpu_median(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy A data to B to avoid changing the original data when sorting
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    thrust::device_vector<float> dev_ptr_sorted(dev_ptr, dev_ptr+A->size);

    // Sort data (see why below)
    thrust::sort(dev_ptr_sorted.begin(), dev_ptr_sorted.end());

    // Get median
    int midpoint = (int)A->size / 2;
    if(A->size % 2==1 && A->size>1) {
         return dev_ptr_sorted[midpoint];
    }else{
        return (dev_ptr_sorted[midpoint-1]+dev_ptr_sorted[midpoint])/2.0f;
    }

}

void gpu_median(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    float *d_aux_ptr;
    check_cuda(cudaMalloc((void**)&(d_aux_ptr), A->size*sizeof(float)),"create map");
    check_cuda(cudaDeviceSynchronize(), "create");

    setDims(B);  // Walk through reduced tensor
    gpu_median<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, d_aux_ptr);
    check_cuda(cudaDeviceSynchronize(),"reduce_median");

    check_cuda(cudaFree(d_aux_ptr),"delete_map");
}


float gpu_var(Tensor *A, bool unbiased){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    float mean = thrust::reduce(dev_ptr, dev_ptr + A->size,0.0f, thrust::plus<float>()) / A->size;
    float sum = thrust::transform_reduce(dev_ptr, dev_ptr + A->size, variance_shift_sum(mean), 0.0f, thrust::plus<float>());

    if(unbiased){return sum/(A->size-1.0f);}
    else {return sum/((float)A->size);}
}

void gpu_var(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_mean<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    gpu_var<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, unbiased);
    check_cuda(cudaDeviceSynchronize(),"reduce_gpu_var");
}

float gpu_std(Tensor *A, bool unbiased){
    return ::sqrtf(gpu_var(A, unbiased));
}

void gpu_std(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_mean<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    gpu_var<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction, unbiased);
    gpu_sqrt<<<dimGrid,dimBlock>>>(B->ptr, B->ptr, B->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_gpu_std");
}


int gpu_mode(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy A data (float) to B (int) for: 1) casting, 2) avoid changing the original data when sorting
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    thrust::device_vector<int> dev_keys(dev_ptr, dev_ptr+A->size);

    // Reserve data for new keys and values
    thrust::device_vector<int> output_keys(A->size);
    thrust::device_vector<int> output_freqs(A->size);

    // Create a tensor fill with ones
    thrust::device_vector<int> dev_ones(A->size);
    thrust::fill(dev_ones.begin(), dev_ones.end(), 1);

    // Sort data (see why below)
    thrust::sort(dev_keys.begin(), dev_keys.end());

    // Reduce contiguous keys: [1 3 3 3 2 2 3] => [1 3 2 1] Vs. [1 3 3 3 3 2 2] => [1 4 2]
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
    new_end = thrust::reduce_by_key(dev_keys.begin(), dev_keys.end(), dev_ones.begin(), output_keys.begin(), output_freqs.begin());

    // Get index of the maximum frequency
    int num_keys = new_end.first  - output_keys.begin();
    thrust::device_vector<int>::iterator iter = thrust::max_element(output_freqs.begin(), output_freqs.begin() + num_keys);
    unsigned int index = iter - output_freqs.begin();

    int most_frequent_key = output_keys[index];
    int most_frequent_val = output_freqs[index];  // Frequencies
    return most_frequent_key;
}

void gpu_mode(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    gpu_initialize_rd(rd, A, B, true);

    setDims(B);  // Walk through reduced tensor
    gpu_mode<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, rd->gpu_addresses, B->size, rd->size_reduction);
    check_cuda(cudaDeviceSynchronize(),"reduce_mode");
}

// GPU: Reduction ***************************
void gpu_sum2D(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC){
    int device=A->gpu_device;
    cudaSetDevice(device);

    int m=A->shape[1];
    int n=B->shape[0];
    int ldA=A->shape[1];
    int ldB=B->shape[1];
    int ldC=A->shape[1];

    float alfa=scA;
    float beta=scB;
    float one=1.0;

    if (incC){
        check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,A->ptr,ldA,&one,C->ptr,ldB,C->ptr,ldC),"sum2D");
        check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,B->ptr,ldA,&one,C->ptr,ldB,C->ptr,ldC),"sum2D");
    }
    else {
        check_cublas(
                cublasSgeam(hcublas[device], CUBLAS_OP_N, CUBLAS_OP_N, m, n, &alfa, A->ptr, ldA, &beta, B->ptr, ldB,
                            C->ptr, ldC), "sum2D");
    }
}


void gpu_initialize_rd(ReduceDescriptor2 *rd, Tensor *A, Tensor *B, bool reverse){
    // TODO: TEMP! I don't like this approach
    if(rd->gpu_addresses == nullptr){
        int size = A->size;

        // Build cpu map (if needed)
        if(rd->cpu_addresses == nullptr){
            rd->build_map(reverse);
        }

        check_cuda(cudaMalloc((void**)&(rd->gpu_addresses), size*sizeof(int)),"create map");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(rd->gpu_addresses, rd->cpu_addresses, size*sizeof(int),cudaMemcpyHostToDevice),"copy map");
        check_cuda(cudaDeviceSynchronize(), "copy");

        // Delete cpu
        if(rd->cpu_addresses != nullptr){
            delete rd->cpu_addresses;
            rd->cpu_addresses = nullptr;
        }
    }
}