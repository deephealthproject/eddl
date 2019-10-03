#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


// CPU: Math (in-place) ********************************************
void gpu_abs_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    abs_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "abs_");
}

void gpu_acos_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    acos_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "acos_");
}

void gpu_add_(Tensor *A, float v) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  add_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, v);
  check_cuda(cudaDeviceSynchronize(), "add_");
}

void gpu_asin_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    asin_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "asin_");
}

void gpu_atan_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    atan_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "atan_");
}

void gpu_ceil_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    ceil_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "ceil_");
}

void gpu_clamp_(Tensor *A, float min, float max){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    clamp_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, min, max);
    check_cuda(cudaDeviceSynchronize(), "clamp_");
}

void gpu_cos_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    cos_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "cos_");
}

void gpu_cosh_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    cosh_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "cosh_");
}

void gpu_exp_(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  exp_<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"exp_");

}

void gpu_floor_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    floor_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "floor_");
}


void gpu_log_(Tensor *A) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  log_<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0], c);
  check_cuda(cudaDeviceSynchronize(), "log_");

}


void gpu_log2_(Tensor *A) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    log2_<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(),"log2_");
}


void gpu_log10_(Tensor *A) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    log10_<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(),"log10_");
}


void gpu_logn_(Tensor *A, float n){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    logn_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, n);
    check_cuda(cudaDeviceSynchronize(), "logn_");
};

void gpu_mod_(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    mod_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, v);
    check_cuda(cudaDeviceSynchronize(), "mod_");
}

void gpu_mult_(Tensor *A, float v) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  mult_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, v);
  check_cuda(cudaDeviceSynchronize(),"mult_");

}

void gpu_normalize_(Tensor *A, float min, float max){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    // TODO: Temp
    float min_ori = 0;
    float max_ori = 10000;

    normalize_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, min_ori, max_ori, min, max);
    check_cuda(cudaDeviceSynchronize(), "normalize_");
}

void gpu_pow_(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    pow_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, v);
    check_cuda(cudaDeviceSynchronize(), "pow_");
}


void gpu_reciprocal_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    reciprocal_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "reciprocal_");
}

void gpu_remainder_(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    remainder_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, v);
    check_cuda(cudaDeviceSynchronize(), "remainder_");
}

void gpu_round_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    round_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "round_");
}

void gpu_rsqrt_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    rsqrt_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "rsqrt_");
}

void gpu_sigmoid_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sigmoid_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "sigmoid_");
}

void gpu_sign_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sign_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "sign_");
}

void gpu_sin_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sin_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "sin_");
}

void gpu_sinh_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    sinh_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "sinh_");
}


void gpu_sqr_(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sqr_<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sqr_");

}

void gpu_sqrt_(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sqrt_<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sqrt_");
}

void gpu_tan_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    tan_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "tan_");
}

void gpu_tanh_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    tanh_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
    check_cuda(cudaDeviceSynchronize(), "tanh_");
}

void gpu_trunc_(Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    trunc_<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c);
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

  el_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->shape[0],r);

  check_cuda(cudaDeviceSynchronize(),"gpu_el_div");
}


void gpu_el_mult(Tensor *A, Tensor *B, Tensor *C,int incC){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  el_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,incC,A->shape[0],c);

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

// CPU: Should be reductions ***************************************

void gpu_total_sum(Tensor *A, float *tot)
{
  float *total;
  int device=A->gpu_device;
  cudaSetDevice(device);
  float t=0;


  setDims(A);

  check_cuda(cudaMalloc((void**)&total,sizeof(float)),"create float in sum");

  check_cuda(cudaMemcpy(total,&t,sizeof(float),cudaMemcpyHostToDevice),"error copy in sum");

  reduce_array_sum<<<dimGrid,dimBlock>>>(A->ptr,A->size,total);

  check_cuda(cudaMemcpy(tot,total,sizeof(float),cudaMemcpyDeviceToHost),"error copy in sum");

  check_cuda(cudaFree(total),"delete float in sum");
}




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
    else
        check_cublas(cublasSgeam(hcublas[device],CUBLAS_OP_N,CUBLAS_OP_N, m,n,&alfa,A->ptr,ldA,&beta,B->ptr,ldB,C->ptr,ldC),"sum2D");

}


void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB){

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  if (!incB) gpu_set(B,0.0);

  reduce_sum2D<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->shape[0],A->shape[1],axis);


  check_cuda(cudaDeviceSynchronize(),"reduce_sum2D");
}
