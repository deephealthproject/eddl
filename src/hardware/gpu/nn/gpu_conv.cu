/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"

#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

void * shared_workspace=nullptr;
size_t workspace_size=0;

void gpu_im2col(ConvolDescriptor *D, int col2im){
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  setDims(D->gpuI)
  dimGrid.x*=D->I->shape[0];

  if (col2im)
    gpu_im2col_k<<<dimGrid,dimBlock>>>(D->ID->ptr, D->gpuI->ptr,D->I->shape[0],D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padrt,D->padrb,D->padcl,D->padcr,1);
  else
    gpu_im2col_k<<<dimGrid,dimBlock>>>(D->I->ptr, D->gpuI->ptr,D->I->shape[0],D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padrt,D->padrb,D->padcl,D->padcr,0);

  check_cuda(cudaDeviceSynchronize(),"gpu_im2col");

}

void gpu_im2col_low(ConvolDescriptor *D, int col2im,int b){
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  setDims(D->gpuI)

  if (col2im)
    gpu_im2col_k_low<<<dimGrid,dimBlock>>>(D->ID->ptr, b, D->gpuI->ptr,D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padrt,D->padrb,D->padcl,D->padcr,1);
  else
    gpu_im2col_k_low<<<dimGrid,dimBlock>>>(D->I->ptr, b, D->gpuI->ptr,D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padrt,D->padrb,D->padcl,D->padcr,0);

  check_cuda(cudaDeviceSynchronize(),"gpu_im2col");

}




void gpu_conv2D(ConvolDescriptor *D) {

  cout<<"STARTTTTTTTTTT   AQUI" <<endl;
  int device=D->I->gpu_device;
  cudaSetDevice(device);
#ifndef cCUDNN
  int osize=D->z*D->r*D->c;
  int isize=D->kz*D->kr*D->kc*D->r*D->c;
  D->gpuK->ptr=D->K->ptr;
  D->gpuO->ptr=D->O->ptr;
  D->gpuI->ptr=D->gpuIB->ptr;


  if (D->mem_level>1) {
    for(int b=0;b<D->I->shape[0];b++,D->gpuO->ptr+=osize) {
      gpu_im2col_low(D,0,b);
      gpu_mult2D(D->gpuK,0,D->gpuI,1,D->gpuO,0);
    }
  }
  else {

    gpu_im2col(D,0);
    if (D->mem_level==0) {
      gpu_mult2D(D->gpuK,0,D->gpuIB,1,D->gpuOB,0);
      setDims(D->O);
      gpu_traspose_batch_depth<<<dimGrid,dimBlock>>>(D->gpuOB->ptr, D->O->ptr, D->O->shape[0], D->z, D->r, D->c);
      check_cuda(cudaDeviceSynchronize(),"gpu_batch_depth");
    }
    else {
      gpu_im2col(D,0);
      for(int b=0;b<D->I->shape[0];b++,D->gpuO->ptr+=osize,D->gpuI->ptr+=isize)
        gpu_mult2D(D->gpuK,0,D->gpuI,1,D->gpuO,0);
    }

  }
#else
  if (D->cudnn_env_init < 0){
      cout<<"AQUI" <<endl;
      D->cudnn_env_init = 1;
      cout<<"AQUI 2" <<endl;
      int requestedAlgoCount;
      cout<<"AQUI 3" <<endl;
      check_cudnn(cudnnGetConvolutionForwardAlgorithmMaxCount(D->cudnn_handle, &requestedAlgoCount));
      cout<<"AQUI 4" <<endl;
      int returnedAlgoCount;
      cout<<"AQUI 5" <<endl;
      cudnnConvolutionFwdAlgoPerf_t * perfResults = new cudnnConvolutionFwdAlgoPerf_t [requestedAlgoCount];
      check_cudnn(cudnnFindConvolutionForwardAlgorithm( D->cudnn_handle, D->xDesc, D->wDesc, D->convolution_descriptor, D->yDesc,
                  requestedAlgoCount, &returnedAlgoCount, perfResults));
      cout<<" A total of "<< returnedAlgoCount <<" where tested and the best is: "<< perfResults[0].algo <<endl;
      D->fwd_algorithm = perfResults[0].algo;
      check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(D->cudnn_handle,D->xDesc, D->wDesc, D->convolution_descriptor,  D->yDesc,
                                                          D->fwd_algorithm, &workspace_size));
      cout<<"It needs "<< workspace_size << "bytes" <<endl;
  }
/*  cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionFwdAlgo_t           algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       yDesc,
    void                               *y);*/
#endif
  if (D->use_bias) {
    int size=D->bias->shape[0];
    for(int i=0;i<size;i+=1024) {
      int s=min(1024,size-i);
      gpu_addbias_k<<<D->O->shape[0],s>>>(D->O->ptr, D->O->shape[0], D->r,D->c,D->nk,D->bias->ptr,i);
      check_cuda(cudaDeviceSynchronize(),"gpu_addbias");
    }
  }



}


void gpu_conv2D_grad(ConvolDescriptor *D){

  int device=D->I->gpu_device;

  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;
  int isize=D->kz*D->kr*D->kc*D->r*D->c;

  D->gpugK->ptr=D->gK->ptr;
  D->gpuD->ptr=D->D->ptr;
  D->gpuI->ptr=D->gpuIB->ptr;

  if (D->mem_level>1) {
    for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize){
      gpu_im2col_low(D,0,b);
      gpu_mult2D(D->gpuD,0,D->gpuI,0,D->gpugK,1);
    }
  }
  else {
    if (D->mem_level==0) {
      setDims(D->D);
      gpu_traspose_batch_depth<<<dimGrid,dimBlock>>>(D->D->ptr, D->gpuOB->ptr, D->z, D->O->shape[0], D->r, D->c);
      check_cuda(cudaDeviceSynchronize(),"gpu_batch_depth");

      gpu_mult2D(D->gpuOB,0,D->gpuIB,0,D->gpugK,1);
    }
    else {
      for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize,D->gpuI->ptr+=isize)
        gpu_mult2D(D->gpuD,0,D->gpuI,0,D->gpugK,1);
    }
  }

  if (D->use_bias) {
    int size=D->bias->shape[0];
    for(int i=0;i<size;i+=1024) {
      int s=min(1024,size-i);
      gpu_deltabias_k<<<D->D->shape[0],s>>>(D->D->ptr, D->D->shape[0], D->r,D->c,D->nk,D->gbias->ptr,i);
      check_cuda(cudaDeviceSynchronize(),"gpu_deltabias");
    }
  }


}


void gpu_conv2D_back(ConvolDescriptor *D){


  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;
  int isize=D->kz*D->kr*D->kc*D->r*D->c;
  D->gpuK->ptr=D->K->ptr;
  D->gpuD->ptr=D->D->ptr;
  D->gpuI->ptr=D->gpuIB->ptr;


  if (D->mem_level>1) {
    for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize) {
        gpu_mult2D(D->gpuD, 1, D->gpuK, 0, D->gpuI, 0);
        gpu_im2col_low(D,1,b);
    }
  }
  else {
    if (D->mem_level==0) {
      setDims(D->D);
      gpu_traspose_batch_depth<<<dimGrid,dimBlock>>>(D->D->ptr, D->gpuOB->ptr,  D->z, D->O->shape[0],D->r, D->c);
      check_cuda(cudaDeviceSynchronize(),"gpu_batch_depth");

      gpu_mult2D(D->gpuOB, 1, D->gpuK, 0, D->gpuIB, 0);
      D->gpuI->ptr=D->gpuIB->ptr;
      gpu_im2col(D,1);
    }
    else{
      for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize,D->gpuI->ptr+=isize) {
          gpu_mult2D(D->gpuD, 1, D->gpuK, 0, D->gpuI, 0);
      }
      D->gpuI->ptr=D->gpuIB->ptr;
      gpu_im2col(D,1);
    }
  }


}
