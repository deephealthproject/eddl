/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
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

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"



void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB){

    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    if (!incB) gpu_fill_(B,0.0);

    reduce_sum2D<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->shape[0],A->shape[1],axis);

    check_cuda(cudaDeviceSynchronize(),"reduce_sum2D");
}


void gpu_reduceTosum(Tensor *A, Tensor *B, int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);

    reduceToSum<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, A->shape[axis]);

    check_cuda(cudaDeviceSynchronize(),"reduceTosum");
}

void gpu_reduction(ReduceDescriptor *RD){
  int device=RD->I->gpu_device;

  cudaSetDevice(device);

  int i,j,d,s,p;


  // [MEAN]: Compute items to be reduced
  if (RD->m==0) {
      d=1;
      for(i=0;i<RD->axis.size();i++){
          d *= RD->I->shape[RD->axis[i]];
      }
  }

  //////// Init
  if (RD->ind==nullptr) {
    RD->red_size=RD->index[0].size();
    s=RD->index.size()*RD->red_size;

    int *ind=(int *)malloc(s*sizeof(int));

    for(i=0;i<RD->index.size();i++) {
      p=i*RD->red_size;
      for(j=0;j<RD->index[i].size();j++,p++)
        ind[p]=RD->index[i][j];
    }

    if (RD->m<2) RD->S=RD->O;

    check_cuda(cudaMalloc((void**)&(RD->ind),s*sizeof(int)),"create_index");
    check_cuda(cudaDeviceSynchronize(), "create ind");

    check_cuda(cudaMemcpy(RD->ind,ind,s*sizeof(int),cudaMemcpyHostToDevice),"copy ind");
    check_cuda(cudaDeviceSynchronize(), "copy");

    check_cuda(cudaMalloc((void**)&(RD->red),RD->index.size()*sizeof(float)),"create_tensor");

    free(ind);
  }
  /////////////

  int fast=0;
  if (RD->factor*RD->index.size()<RD->red_size) fast=1;


  if ((fast)&&((RD->m==0)&&(RD->keepdims))) {//mean with keepdims=true (BN)

    setDims(RD->O);
    reduction_permute<<<dimGrid,dimBlock>>>(RD->I->ptr, RD->O->ptr, RD->ind, RD->O->size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");

    for(int i=0;i<RD->index.size();i++) {
      float *ptr=RD->O->ptr+(i*RD->red_size);

      thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(ptr);
      thrust::device_ptr<float> base = thrust::device_pointer_cast(RD->red);

      float sum = thrust::reduce(dev_ptr, dev_ptr + RD->red_size);
      thrust::fill(base + i, base + i + 1, (float)sum/RD->red_size);
    }

    reduction_kernel_keep<<<dimGrid,dimBlock>>>(RD->red, RD->O->ptr,RD->ind, RD->index.size(),RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");

  }else{ // still slow for max, min on conv
    RD->O->fill_(0.0);
    dim3 dimGrid(RD->index.size());
    dim3 dimBlock(1);
    reduction_kernel<<<dimGrid,dimBlock>>>(RD->I->ptr, RD->O->ptr, RD->S->ptr,RD->m, RD->keepdims,d,RD->ind,RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
  }

}


void gpu_reduction_back(ReduceDescriptor *RD){
  int device=RD->I->gpu_device;

  cudaSetDevice(device);

  int d,i;

  // [MEAN]: Compute items to be reduced
  if (RD->m==0) {
      d=1;
      for(i=0;i<RD->axis.size();i++){
          d *= RD->I->shape[RD->axis[i]];
      }
  }

  int fast=0;
  if (RD->factor*RD->index.size()<RD->red_size) fast=1;

  if ((fast)&&((RD->m==0)&&(RD->keepdims))) {// mean with keepdims=true (BN)
    float *aux;
    check_cuda(cudaMalloc((void**)&aux,RD->D->size*sizeof(float)),"create_tensor");

    setDims(RD->D);
    reduction_permute<<<dimGrid,dimBlock>>>(RD->D->ptr, aux, RD->ind, RD->O->size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");

    for(int i=0;i<RD->index.size();i++) {
      float *ptr=aux+(i*RD->red_size);

      thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(ptr);
      thrust::device_ptr<float> base = thrust::device_pointer_cast(RD->red);

      float sum = thrust::reduce(dev_ptr, dev_ptr + RD->red_size);
      thrust::fill(base+i, base + i + 1, (float)sum/RD->red_size);
    }

    check_cuda(cudaFree(aux),"delete_tensor");

    reduction_kernel_keep_inc<<<dimGrid,dimBlock>>>(RD->red, RD->ID->ptr, RD->ind, RD->index.size(),RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");

  }else{ // still slow for max, min on conv
    dim3 dimGrid(RD->index.size());
    dim3 dimBlock(1);
    reduction_back_kernel<<<dimGrid,dimBlock>>>(RD->D->ptr, RD->ID->ptr, RD->S->ptr,RD->m, RD->keepdims,d,RD->ind,RD->red_size);
    check_cuda(cudaDeviceSynchronize(), "reduction_kernel");
  }
}
