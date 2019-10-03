#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


void gpu_rand_uniform(Tensor *A, float v){
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_uniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

  gpu_mult(A,v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

}


void gpu_rand_signed_uniform(Tensor *A, float v){
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_signed_uniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_signed_uniform");

  gpu_mult(A,2*v);
  gpu_add(A,-v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_signed_uniform");

}


void gpu_rand_normal(Tensor *A, float m, float s){
  int device=A->gpu_device;
  cudaSetDevice(device);

  if (A->size%2) {
    gpu_set(A,0.0);
    check_curand(curandGenerateNormal(random_generator[device],A->ptr,A->size-1,m,s),"gpu_rand_normal");
  }
  else
    check_curand(curandGenerateNormal(random_generator[device],A->ptr,A->size,m,s),"gpu_rand_normal");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_normal");

}


void gpu_rand_binary(Tensor *A, float v){
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_binary");

  gpu_mask(A,v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_binary");

}

