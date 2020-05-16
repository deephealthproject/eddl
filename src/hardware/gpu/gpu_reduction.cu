/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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

#include <stdexcept>

#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"
#include "eddl/hardware/gpu/gpu_hw.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

void gpu_reduce(Tensor *A, Tensor *B,string mode,int* map)
{
  int device=A->gpu_device;
  int *gmap;

  cudaSetDevice(device);

  int s=A->size/B->size;


  check_cuda(cudaMalloc((void**)&(gmap),A->size*sizeof(int)),"create map");
  check_cuda(cudaDeviceSynchronize(), "create");

  check_cuda(cudaMemcpy(gmap,map,A->size*sizeof(int),cudaMemcpyHostToDevice),"copy map");
  check_cuda(cudaDeviceSynchronize(), "copy");

  if (mode=="mean") {
    B->fill_(0.0);

    setDims(A);
    reduce_mean<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");

    B->div_(s);

  }
  else if (mode=="variance") {
    Tensor *C=new Tensor(B->getShape(),B->device);

    C->fill_(0.0);

    setDims(A);
    reduce_mean<<<dimGrid,dimBlock>>>(A->ptr,C->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");

    C->div_(s);

    gpu_reduce_op(A,C,"diff",map);

    A->sqr_();

    B->fill_(0.0);

    reduce_mean<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean_var");

    B->div_(s);

    delete C;

  }
  else {
    throw std::invalid_argument("mode: " + mode + " not yet implemented");
  }


  check_cuda(cudaFree(gmap),"delete_map");

}

void gpu_reduce(Tensor *A, Tensor *B,string mode,MapReduceDescriptor *MD)
{
  int device=A->gpu_device;

  cudaSetDevice(device);

  int s=A->size/B->size;

  if (MD->gind==nullptr) {
    check_cuda(cudaMalloc((void**)&(MD->gind),A->size*sizeof(int)),"create map");
    check_cuda(cudaDeviceSynchronize(), "create");

    check_cuda(cudaMemcpy(MD->gind,MD->ind,A->size*sizeof(int),cudaMemcpyHostToDevice),"copy map");
    check_cuda(cudaDeviceSynchronize(), "copy");
  }

  if (mode=="mean") {
    B->fill_(0.0);

    setDims(A);
    reduce_mean<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");

    B->div_(s);

  }
  else if (mode=="variance") {
    Tensor *C=new Tensor(B->getShape(),B->device);

    C->fill_(0.0);

    setDims(A);
    reduce_mean<<<dimGrid,dimBlock>>>(A->ptr,C->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");

    C->div_(s);

    gpu_reduce_op(A,C,"diff",MD);

    A->sqr_();

    B->fill_(0.0);

    reduce_mean<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean_var");

    B->div_(s);

    delete C;

  }
  else {
    throw std::invalid_argument("mode: " + mode + " not yet implemented");
  }

}


void gpu_reduce_op(Tensor *A, Tensor *B,string op,int *map)
{
  int device=A->gpu_device;
  int *gmap;

  cudaSetDevice(device);

  check_cuda(cudaMalloc((void**)&(gmap),A->size*sizeof(int)),"create map");
  check_cuda(cudaDeviceSynchronize(), "create");

  check_cuda(cudaMemcpy(gmap,map,A->size*sizeof(int),cudaMemcpyHostToDevice),"copy map");
  check_cuda(cudaDeviceSynchronize(), "copy");

  if (op=="sum") {
    setDims(A);
    reduce_op_sum<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else if (op=="diff") {
    setDims(A);
    reduce_op_diff<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else if (op=="mult") {
    setDims(A);
    reduce_op_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else if (op=="div") {
    setDims(A);
    reduce_op_div<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,gmap,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else {
    throw std::invalid_argument("op: " + op + " not yet implemented");
  }


  check_cuda(cudaFree(gmap),"delete_map");

}


void gpu_reduce_op(Tensor *A, Tensor *B,string op,MapReduceDescriptor *MD)
{
  int device=A->gpu_device;

  cudaSetDevice(device);

  if (MD->gind==nullptr) {
    check_cuda(cudaMalloc((void**)&(MD->gind),A->size*sizeof(int)),"create map");
    check_cuda(cudaDeviceSynchronize(), "create");

    check_cuda(cudaMemcpy(MD->gind,MD->ind,A->size*sizeof(int),cudaMemcpyHostToDevice),"copy map");
    check_cuda(cudaDeviceSynchronize(), "copy");
  }

  if (op=="sum") {
    setDims(A);
    reduce_op_sum<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else if (op=="diff") {
    setDims(A);
    reduce_op_diff<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else if (op=="mult") {
    setDims(A);
    reduce_op_mult<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else if (op=="div") {
    setDims(A);
    reduce_op_div<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,MD->gind,A->size);
    check_cuda(cudaDeviceSynchronize(),"reduce_mean");
  }
  else {
    throw std::invalid_argument("op: " + op + " not yet implemented");
  }

}


void gpu_reduce_sum2D(Tensor *A,Tensor *B,int axis,int incB){

    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    if (!incB) gpu_fill_(B,0.0);

    reduce_sum2D<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->shape[0],A->shape[1],axis);

    check_cuda(cudaDeviceSynchronize(),"reduce_sum2D");
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
