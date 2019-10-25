/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es), (jmaronasm@gmail.com)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <thrust/device_ptr.h>
//#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

// GPU: Structs for Thrust ********************************************

struct sum_abs_value : public thrust::unary_function<float, float>
{
    __host__ __device__ float operator()(const float &x) const
    {
        return fabsf(x);
    }
};

// GPU: Math (in-place) ********************************************
void gpu_abs_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    abs_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "abs_");
}

void gpu_acos_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    acos_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "acos_");
}

void gpu_add_(Tensor *A, float v) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  add_<<<dimGrid,dimBlock>>>(A->ptr, A->size, v);
  check_cuda(cudaDeviceSynchronize(), "add_");
}

void gpu_asin_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    asin_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "asin_");
}

void gpu_atan_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    atan_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "atan_");
}

void gpu_ceil_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    ceil_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "ceil_");
}

void gpu_clamp_(Tensor *A, float min, float max){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    clamp_<<<dimGrid,dimBlock>>>(A->ptr, A->size, min, max);
    check_cuda(cudaDeviceSynchronize(), "clamp_");
}

void gpu_cos_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    cos_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "cos_");
}

void gpu_cosh_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    cosh_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "cosh_");
}

void gpu_exp_(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  exp_<<<dimGrid,dimBlock>>>(A->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"exp_");

}

void gpu_floor_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    floor_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "floor_");
}


void gpu_log_(Tensor *A) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  log_<<<dimGrid,dimBlock>>>(A->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(), "log_");

}


void gpu_log2_(Tensor *A) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    log2_<<<dimGrid,dimBlock>>>(A->ptr,A->size);
    check_cuda(cudaDeviceSynchronize(),"log2_");
}


void gpu_log10_(Tensor *A) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    log10_<<<dimGrid,dimBlock>>>(A->ptr,A->size);
    check_cuda(cudaDeviceSynchronize(),"log10_");
}


void gpu_logn_(Tensor *A, float n){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    logn_<<<dimGrid,dimBlock>>>(A->ptr, A->size, n);
    check_cuda(cudaDeviceSynchronize(), "logn_");
};

void gpu_mod_(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    mod_<<<dimGrid,dimBlock>>>(A->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "mod_");
}

void gpu_inv_(Tensor *A)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  inv_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
  check_cuda(cudaDeviceSynchronize(),"inv_");

}

void gpu_mult_(Tensor *A, float v) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  mult_<<<dimGrid,dimBlock>>>(A->ptr, A->size, v);
  check_cuda(cudaDeviceSynchronize(),"mult_");

}

void gpu_normalize_(Tensor *A, float min, float max){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    float min_ori = gpu_min(A);
    float max_ori = gpu_max(A);

    normalize_<<<dimGrid,dimBlock>>>(A->ptr, A->size, min_ori, max_ori, min, max);
    check_cuda(cudaDeviceSynchronize(), "normalize_");
}

void gpu_pow_(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    pow_<<<dimGrid,dimBlock>>>(A->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "pow_");
}


void gpu_reciprocal_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    reciprocal_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "reciprocal_");
}

void gpu_remainder_(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    remainder_<<<dimGrid,dimBlock>>>(A->ptr, A->size, v);
    check_cuda(cudaDeviceSynchronize(), "remainder_");
}

void gpu_round_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    round_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "round_");
}

void gpu_rsqrt_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    rsqrt_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "rsqrt_");
}

void gpu_sigmoid_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sigmoid_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sigmoid_");
}

void gpu_sign_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sign_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sign_");
}

void gpu_sin_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sin_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sin_");
}

void gpu_sinh_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sinh_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "sinh_");
}


void gpu_sqr_(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sqr_<<<dimGrid,dimBlock>>>(A->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"sqr_");

}

void gpu_sqrt_(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sqrt_<<<dimGrid,dimBlock>>>(A->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"sqrt_");
}

void gpu_tan_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    tan_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "tan_");
}

void gpu_tanh_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    tanh_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "tanh_");
}

void gpu_trunc_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    trunc_<<<dimGrid,dimBlock>>>(A->ptr, A->size);
    check_cuda(cudaDeviceSynchronize(), "trunc_");
}

// CPU: Math (static) ********************************************


void gpu_addc(float scA,Tensor *A, float scB,Tensor *B, Tensor *C,int incC){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);


    addc<<<dimGrid,dimBlock>>>(scA,A->ptr,scB,B->ptr,C->ptr,incC,A->size);
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


void gpu_el_div(Tensor *A, Tensor *B, Tensor *C,int incC) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  el_div<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->size);

  check_cuda(cudaDeviceSynchronize(),"gpu_el_div");
}


void gpu_el_mult(Tensor *A, Tensor *B, Tensor *C,int incC){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  el_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->size);

  check_cuda(cudaDeviceSynchronize(),"gpu_el_mult");
}


void gpu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);


  sum_mat_row<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}


void gpu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sum_mat_col<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->shape[0],A->shape[1]);

  check_cuda(cudaDeviceSynchronize(),"sum2D_rowwise");

}


// GPU: Should be reductions ***************************
float gpu_max(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return *thrust::max_element(dev_ptr, dev_ptr + A->size);
}

float gpu_min(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return *thrust::min_element(dev_ptr, dev_ptr + A->size);
}


float gpu_sum(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
    return thrust::reduce(dev_ptr, dev_ptr + A->size);
}

float gpu_sum_abs(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(A->ptr);
//    return thrust::transform_reduce(dev_ptr, dev_ptr + A->size, thrust::plus<float>, 0.0f, sum_abs_value);
    return 0.0f;
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
