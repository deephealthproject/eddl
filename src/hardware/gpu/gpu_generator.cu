/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es), (jmaronasm@gmail.com)
* All rights reserved
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"



float* gpu_get_uniforms(int N){
    /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, N * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<N, 1>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  float* gpu_nums;
  cudaMalloc((void**) &gpu_nums, N * sizeof(float));

  /* invoke the kernel to get some random numbers */
  random_uniform<<<N, 1>>>(states, gpu_nums);

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  // cudaFree(gpu_nums);
  return gpu_nums;
}


void gpu_rand_uniform(Tensor *A, float v){
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_uniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

  //gpu_mult_(A, v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_uniform");

}


void gpu_rand_signed_uniform(Tensor *A, float v){
  int device=A->gpu_device;
  cudaSetDevice(device);

  check_curand(curandGenerateUniform(random_generator[device],A->ptr,A->size),"gpu_rand_signed_uniform");

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_signed_uniform");

  gpu_mult_(A, 2*v);
  gpu_add_(A, -v);

  check_cuda(cudaDeviceSynchronize(),"gpu_rand_signed_uniform");

}


void gpu_rand_normal(Tensor *A, float m, float s){
  int device=A->gpu_device;
  cudaSetDevice(device);

  if (A->size%2) {
    gpu_fill_(A, 0.0);
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
