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

#ifdef cCUDNN
void * shared_workspace=nullptr;
size_t workspace_size=0;

void my_get_fdescriptor(cudnnFilterDescriptor_t t, char * name){

    cudnnDataType_t         dataType;
    cudnnTensorFormat_t       format;
    int                     n;
    int                     c;
    int                     h;
    int                     w;
    check_cudnn(cudnnGetFilter4dDescriptor(t, &dataType, &format, &n, &c, &h, &w));
    std::cout<<name<<": ("<<dataType<<", "<<n<<", "<<c<<", "<<h<<", "<<w<<")"<<std::endl;
}

void my_get_descriptor(cudnnTensorDescriptor_t t, char * name){

    cudnnDataType_t         dataType;
    int                     n;
    int                     c;
    int                     h;
    int                     w;
    int                     nStride;
    int                     cStride;
    int                     hStride;
    int                     wStride;
    check_cudnn(cudnnGetTensor4dDescriptor(t, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride));
    std::cout<<name<<": ("<<dataType<<", "<<n<<", "<<c<<", "<<h<<", "<<w<<", "<<nStride<<", "<<cStride<<", "<<hStride<<", "<<wStride<<")"<<std::endl;
}



int allocate_workspace(size_t size){
    if (size <= workspace_size){
        return 0;
    }
    else {
        workspace_size = size;
        cudaFree(shared_workspace);
        return cudaMalloc((void **) &shared_workspace, size);
    }
}
#endif

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

  int device=D->I->gpu_device;
  cudaSetDevice(device);
  float alpha = 1.0f;
  float beta = 0.0f;

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
  // FWD environment
  if (D->cudnn_env_init < 0){
      D->cudnn_env_init = 1;

      int requestedAlgoCount;
      check_cudnn(cudnnGetConvolutionForwardAlgorithmMaxCount( D->cudnn_handle, &requestedAlgoCount));

      int returnedAlgoCount;
      cudnnConvolutionFwdAlgoPerf_t * perfResults = new cudnnConvolutionFwdAlgoPerf_t [requestedAlgoCount];

      check_cudnn(cudnnFindConvolutionForwardAlgorithm( D->cudnn_handle, D->xDesc, D->wDesc, D->convolution_descriptor, D->yDesc,
                  requestedAlgoCount, &returnedAlgoCount, perfResults));

      int aux_alg = 0;
      size_t size;
      do{
          D->fwd_algorithm = perfResults[aux_alg].algo;

          check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(D->cudnn_handle,D->xDesc, D->wDesc,
                                                              D->convolution_descriptor,  D->yDesc,
                                                              D->fwd_algorithm, &size));
          aux_alg++;
      }
      while(allocate_workspace(size));
  }
  //BWD environment
  if (D->cudnn_conv_back_init < 0){
      D->cudnn_conv_back_init = 1;
       int requestedAlgoCount;

      check_cudnn(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
              D->cudnn_handle, &requestedAlgoCount));
      int returnedAlgoCount;
      cudnnConvolutionBwdFilterAlgoPerf_t * perfResults = new cudnnConvolutionBwdFilterAlgoPerf_t [requestedAlgoCount];

      check_cudnn(cudnnFindConvolutionBackwardFilterAlgorithm(D->cudnn_handle, D->xDesc, D->yDesc,
                                                        D->convolution_descriptor, D->wDesc, requestedAlgoCount,
                                                        &returnedAlgoCount, perfResults));
      int aux_alg = 0;
      size_t size;
      do{
          D->bwd_filter_algorithm = perfResults[aux_alg].algo;

          check_cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(D->cudnn_handle,D->xDesc, D->yDesc,
                                                              D->convolution_descriptor,  D->wDesc,
                                                              D->bwd_filter_algorithm, &size));
          aux_alg++;
      }
      while(allocate_workspace(size));

      //////////// DATA!!!!
      requestedAlgoCount = 0;
     check_cudnn(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(D->cudnn_handle, &requestedAlgoCount));
     returnedAlgoCount=0;
      cudnnConvolutionBwdDataAlgoPerf_t * perfResults_d = new cudnnConvolutionBwdDataAlgoPerf_t [requestedAlgoCount];

      check_cudnn(cudnnFindConvolutionBackwardDataAlgorithm(D->cudnn_handle, D->wDesc, D->yDesc,
                                                        D->convolution_descriptor, D->xDesc, requestedAlgoCount,
                                                        &returnedAlgoCount, perfResults_d));
      aux_alg = 0;
       size=0;
      do{
          D->bwd_data_algorithm = perfResults_d[aux_alg].algo;

          check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(D->cudnn_handle,D->wDesc, D->yDesc,
                                                              D->convolution_descriptor,  D->xDesc,
                                                              D->bwd_data_algorithm, &size));
          aux_alg++;
      }
      while(allocate_workspace(size));

  }
  check_cudnn(cudnnConvolutionForward( D->cudnn_handle, &alpha, D->xDesc, D->I->ptr,
                                       D->wDesc, D->K->ptr,
                                       D->convolution_descriptor, D->fwd_algorithm,
                                       shared_workspace, workspace_size,
                                       &beta, D->yDesc, D->O->ptr));
#endif
  if (D->use_bias) {
#ifndef cCUDNN
    int size=D->bias->shape[0];
    for(int i=0;i<size;i+=1024) {
      int s=min(1024,size-i);
      gpu_addbias_k<<<D->O->shape[0],s>>>(D->O->ptr, D->O->shape[0], D->r,D->c,D->nk,D->bias->ptr,i);
      check_cuda(cudaDeviceSynchronize(),"gpu_addbias");
    }
#else
    check_cudnn(cudnnAddTensor(D->cudnn_handle, &alpha, D->bDesc, D->bias->ptr,
                               &alpha, D->yDesc, D->O->ptr));
#endif
  }


}


void gpu_conv2D_grad(ConvolDescriptor *D){

  int device=D->I->gpu_device;

  cudaSetDevice(device);
  float alpha=1.0;
  float beta = 0.0;
#ifndef cCUDNN
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
#else
        check_cudnn(cudnnConvolutionBackwardFilter(D->cudnn_handle, &alpha,
                                      D->xDesc, D->I->ptr,
                                      D->yDesc, D->D->ptr, D->convolution_descriptor,
                                      D->bwd_filter_algorithm,
                                      shared_workspace, workspace_size,
                                      &beta, D->wDesc, D->gK->ptr));

#endif
  if (D->use_bias) {
#ifndef cCUDNN
    int size=D->bias->shape[0];
    for(int i=0;i<size;i+=1024) {
      int s=min(1024,size-i);
      gpu_deltabias_k<<<D->D->shape[0],s>>>(D->D->ptr, D->D->shape[0], D->r,D->c,D->nk,D->gbias->ptr,i);
      check_cuda(cudaDeviceSynchronize(),"gpu_deltabias");
    }
#else
      check_cudnn(cudnnConvolutionBackwardBias(D->cudnn_handle, &alpha, D->yDesc, D->D->ptr,
                                               &beta, D->bDesc, D->gbias->ptr));
#endif

  }


}


void gpu_conv2D_back(ConvolDescriptor *D){


  int device=D->I->gpu_device;
  cudaSetDevice(device);
#ifndef cCUDNN
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
#else
    float alpha = 1.0f;
    float beta = 1.0f;
    check_cudnn(cudnnConvolutionBackwardData(D->cudnn_handle, &alpha, D->wDesc, D->K->ptr,
                                             D->yDesc, D->D->ptr,
                                             D->convolution_descriptor, D->bwd_data_algorithm,
                                             shared_workspace, workspace_size,
                                             &beta, D->xDesc, D->ID->ptr));
#endif

}


void gpu_conv3D(ConvolDescriptor3D *D){

}

void gpu_conv3D_grad(ConvolDescriptor3D *D){

}

void gpu_conv3D_back(ConvolDescriptor3D *D){

}
