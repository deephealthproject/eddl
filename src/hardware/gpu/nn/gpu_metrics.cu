/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "hardware/gpu/gpu_nn.h"
#include "hardware/gpu/gpu_nn_kernels.h"

#include "hardware/gpu/gpu_hw.h"
#include "hardware/gpu/gpu_tensor.h"
#include "hardware/gpu/gpu_kernels.h"

#include "tensor/tensor.h"
#include "descriptors/descriptors.h"


void gpu_accuracy(Tensor *A,Tensor *B,int *acc){
  int device=A->gpu_device;
  cudaSetDevice(device);
  int r,c;

  r=A->shape[0];
  c=A->size/r;

  dim3 dimGrid(r);
  dim3 dimBlock(c);

  float* max_row=gpu_create_tensor(device,r);

  int *a;
  check_cuda(cudaMalloc((void**)&a,sizeof(int)),"error cudaMalloc in accuracy");
  cudaMemset(a, 0, sizeof(int));

  accuracy<<<dimBlock,dimGrid>>>(A->ptr,B->ptr,max_row,c,r,a);
  check_cuda(cudaMemcpy(acc,a,sizeof(int),cudaMemcpyDeviceToHost),"error copy in accuracy");

  cudaFree(a);
  gpu_delete_tensor(device,max_row);

}
