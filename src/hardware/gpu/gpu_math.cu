#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "tensor_cuda.h"
#include "tensor_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


// CPU: Math (in-place) ********************************************

void gpu_add(Tensor *A, float v) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  add<<<dimGrid,dimBlock>>>(A->ptr, v, A->shape[0], c);
  check_cuda(cudaDeviceSynchronize(),"add");

}


void gpu_exp(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  exp<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"exp");

}


void gpu_log(Tensor *A) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  log<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"log");

}


void gpu_log2(Tensor *A) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    log2<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
    check_cuda(cudaDeviceSynchronize(),"log2");
}


void gpu_log10(Tensor *A) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    log10<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
    check_cuda(cudaDeviceSynchronize(),"log10");
}


void gpu_logn(Tensor *A, float n){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    logn<<<dimGrid,dimBlock>>>(A->ptr, A->shape[0], c, n);
    check_cuda(cudaDeviceSynchronize(),"logn");
};


void gpu_mult(Tensor *A, float v) {
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  mult<<<dimGrid,dimBlock>>>(A->ptr,v,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"mult");

}


void gpu_pow(Tensor *A, float v){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    pow<<<dimGrid,dimBlock>>>(A->ptr, v, A->shape[0],c);
    check_cuda(cudaDeviceSynchronize(), "pow");
}


void gpu_sqr(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sqr<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sqr");

}

void gpu_sqrt(Tensor *A) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sqrt<<<dimGrid,dimBlock>>>(A->ptr,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"sqrt");

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
