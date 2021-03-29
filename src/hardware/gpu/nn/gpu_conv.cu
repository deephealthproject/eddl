/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
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
#define cuDNN_GPUS 8
void * shared_workspace[cuDNN_GPUS]; 
size_t workspace_size[cuDNN_GPUS]={0,0,0,0,0,0,0,0};

int allocate_workspace(size_t size, int dev){
    if (size <= workspace_size[dev]){
        return 0;
    }
    else {
        workspace_size[dev] = size;
        cudaFree(shared_workspace[dev]);
        return cudaMalloc((void **) &shared_workspace[dev], size);
    }
}

//Template created for ConvolDescriptor and ConvolDescriptor3D
template <class condesc>
void cuDNN_environment_initialization(condesc *D){

  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int requestedAlgoCount;
  check_cudnn(cudnnGetConvolutionForwardAlgorithmMaxCount( hdnn[device], &requestedAlgoCount),
                                                                "cudnnGetConvolutionForwardAlgorithmMaxCount",__FILE__);

  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t * perfResults = new cudnnConvolutionFwdAlgoPerf_t [requestedAlgoCount];
  check_cudnn(cudnnFindConvolutionForwardAlgorithm( hdnn[device], D->xDesc, D->wDesc, D->convolution_descriptor, D->yDesc,
              requestedAlgoCount, &returnedAlgoCount, perfResults),"cudnnFindConvolutionForwardAlgorithm",__FILE__);

  int aux_alg = 0;
  size_t size;
  do{
      D->fwd_algorithm = perfResults[aux_alg].algo;

      check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(hdnn[device],D->xDesc, D->wDesc,
                                                              D->convolution_descriptor,  D->yDesc,
                                                              D->fwd_algorithm, &size),
                                                        "cudnnGetConvolutionForwardWorkspaceSize",__FILE__);
      aux_alg++;
  }
  while(allocate_workspace(size,device));
  //BWD environment
  requestedAlgoCount = 0;

  check_cudnn(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
              hdnn[device], &requestedAlgoCount),"cudnnGetConvolutionBackwardFilterAlgorithmMaxCount",__FILE__);
  returnedAlgoCount = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t * perfResultsbwf = new cudnnConvolutionBwdFilterAlgoPerf_t [requestedAlgoCount];

  check_cudnn(cudnnFindConvolutionBackwardFilterAlgorithm(hdnn[device], D->xDesc, D->yDesc,
                                                        D->convolution_descriptor, D->wDesc, requestedAlgoCount,
                                                        &returnedAlgoCount, perfResultsbwf),
                                                        "cudnnFindConvolutionBackwardFilterAlgorithm",__FILE__);
  aux_alg = 0;
  size = 0;
  do{
     D->bwd_filter_algorithm = perfResultsbwf[aux_alg].algo;

    check_cudnn(cudnnGetConvolutionBackwardFilterWorkspaceSize(hdnn[device],D->xDesc, D->yDesc,
                                                              D->convolution_descriptor,  D->wDesc,
                                                              D->bwd_filter_algorithm, &size),
                                                  "cudnnGetConvolutionBackwardFilterWorkspaceSize",__FILE__);
    aux_alg++;
  }
   while(allocate_workspace(size,device));

  check_cudnn(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(hdnn[device], &requestedAlgoCount),
                    "cudnnGetConvolutionBackwardDataAlgorithmMaxCount", __FILE__);
  returnedAlgoCount=0;
  cudnnConvolutionBwdDataAlgoPerf_t * perfResults_d = new cudnnConvolutionBwdDataAlgoPerf_t [requestedAlgoCount];

  check_cudnn(cudnnFindConvolutionBackwardDataAlgorithm(hdnn[device], D->wDesc, D->yDesc,
                                                        D->convolution_descriptor, D->xDesc, requestedAlgoCount,
                                                        &returnedAlgoCount, perfResults_d),
                                             "(cudnnFindConvolutionBackwardDataAlgorithm",__FILE__);
  aux_alg = 0;
  size=0;
  do{
      D->bwd_data_algorithm =  perfResults_d[aux_alg].algo;

      check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(hdnn[device],D->wDesc, D->yDesc,
                                                              D->convolution_descriptor,  D->xDesc,
                                                              D->bwd_data_algorithm, &size),
                                             "cudnnGetConvolutionBackwardDataWorkspaceSize",__FILE__);
      aux_alg++;
  }
  while(allocate_workspace(size,device));


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
  float alpha = 1.0f;
  float beta = 0.0f;
  if (D->cudnn_env_init < 0){
      D->cudnn_env_init = 1;
      cuDNN_environment_initialization<ConvolDescriptor>(D);
  }
  check_cudnn(cudnnConvolutionForward( hdnn[device], &alpha, D->xDesc, D->I->ptr,
                                       D->wDesc, D->K->ptr,
                                       D->convolution_descriptor, D->fwd_algorithm,
                                       shared_workspace[device], workspace_size[device],
                                       &beta, D->yDesc, D->O->ptr),"cudnnConvolutionForward",__FILE__);
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
    check_cudnn(cudnnAddTensor(hdnn[device], &alpha, D->bDesc, D->bias->ptr,
                               &alpha, D->yDesc, D->O->ptr),"cudnnAddTensor",__FILE__);
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
        check_cudnn(cudnnConvolutionBackwardFilter(hdnn[device], &alpha,
                                      D->xDesc, D->I->ptr,
                                      D->yDesc, D->D->ptr, D->convolution_descriptor,
                                      D->bwd_filter_algorithm,
                                      shared_workspace[device], workspace_size[device],
                                      &beta, D->wDesc, D->gK->ptr),"cudnnConvolutionBackwardFilter",__FILE__);

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
      check_cudnn(cudnnConvolutionBackwardBias(hdnn[device], &alpha, D->yDesc, D->D->ptr,
                                               &beta, D->bDesc, D->gbias->ptr),"cudnnConvolutionBackwardBias",__FILE__);
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
    float beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(hdnn[device], &alpha, D->wDesc, D->K->ptr,
                                             D->yDesc, D->D->ptr,
                                             D->convolution_descriptor, D->bwd_data_algorithm,
                                             shared_workspace[device], workspace_size[device],
                                             &beta, D->xDesc, D->ID->ptr),"cudnnConvolutionBackwardData",__FILE__);
#endif

}


void gpu_conv3D(ConvolDescriptor3D *D){
 int device=D->I->gpu_device;
  cudaSetDevice(device);
#ifdef cCUDNN
 // FWD environment
  float alpha = 1.0f;
  float beta = 0.0f;
  if (D->cudnn_env_init < 0){
      D->cudnn_env_init = 1;
      cuDNN_environment_initialization<ConvolDescriptor3D>(D);
  }

 check_cudnn(cudnnConvolutionForward( hdnn[device], &alpha, D->xDesc, D->I->ptr,
                                       D->wDesc, D->K->ptr,
                                       D->convolution_descriptor, D->fwd_algorithm,
                                       shared_workspace[device], workspace_size[device],
                                       &beta, D->yDesc, D->O->ptr),"cudnnConvolutionForward",__FILE__);
  if (D->use_bias) {
    check_cudnn(cudnnAddTensor(hdnn[device], &alpha, D->bDesc, D->bias->ptr,
                               &alpha, D->yDesc, D->O->ptr),"cudnnAddTensor",__FILE__);
  }


#endif

}

void gpu_conv3D_grad(ConvolDescriptor3D *D){
 int device=D->I->gpu_device;
  cudaSetDevice(device);
#ifdef cCUDNN
        float alpha = 1.0f;
        float beta = 0.0f;
        check_cudnn(cudnnConvolutionBackwardFilter(hdnn[device], &alpha,
                                      D->xDesc, D->I->ptr,
                                      D->yDesc, D->D->ptr, D->convolution_descriptor,
                                      D->bwd_filter_algorithm,
                                      shared_workspace[device], workspace_size[device],
                                      &beta, D->wDesc, D->gK->ptr),"cudnnConvolutionBackwardFilter",__FILE__);
  if (D->use_bias) {
      check_cudnn(cudnnConvolutionBackwardBias(hdnn[device], &alpha, D->yDesc, D->D->ptr,
                                               &beta, D->bDesc, D->gbias->ptr),"cudnnConvolutionBackwardBias",__FILE__);
   }
#endif

}

void gpu_conv3D_back(ConvolDescriptor3D *D){
 int device=D->I->gpu_device;
  cudaSetDevice(device);
#ifdef cCUDNN
    float alpha = 1.0f;
    float beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(hdnn[device], &alpha, D->wDesc, D->K->ptr,
                                             D->yDesc, D->D->ptr,
                                             D->convolution_descriptor, D->bwd_data_algorithm,
                                             shared_workspace[device], workspace_size[device],
                                             &beta, D->xDesc, D->ID->ptr),"cudnnConvolutionBackwardData",__FILE__);
#endif


}




void gpu_conv2DT(ConvolDescriptorT *D) {

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
    // FWD environment
    float alpha = 1.0f;
    float beta = 0.0f;
    if (D->cudnn_env_init < 0){
        D->cudnn_env_init = 1;
        cuDNN_environment_initialization<ConvolDescriptorT>(D);
    }
    check_cudnn(cudnnConvolutionForward( hdnn[device], &alpha, D->xDesc, D->I->ptr,
                                         D->wDesc, D->K->ptr,
                                         D->convolution_descriptor, D->fwd_algorithm,
                                         shared_workspace[device], workspace_size[device],
                                         &beta, D->yDesc, D->O->ptr),"cudnnConvolutionForward",__FILE__);
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
        check_cudnn(cudnnAddTensor(hdnn[device], &alpha, D->bDesc, D->bias->ptr,
                                   &alpha, D->yDesc, D->O->ptr),"cudnnAddTensor",__FILE__);
#endif
    }


}


void gpu_conv2DT_grad(ConvolDescriptorT *D){

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
    check_cudnn(cudnnConvolutionBackwardFilter(hdnn[device], &alpha,
                                               D->xDesc, D->I->ptr,
                                               D->yDesc, D->D->ptr, D->convolution_descriptor,
                                               D->bwd_filter_algorithm,
                                               shared_workspace[device], workspace_size[device],
                                               &beta, D->wDesc, D->gK->ptr),"cudnnConvolutionBackwardFilter",__FILE__);

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
        check_cudnn(cudnnConvolutionBackwardBias(hdnn[device], &alpha, D->yDesc, D->D->ptr,
                                                 &beta, D->bDesc, D->gbias->ptr),"cudnnConvolutionBackwardBias",__FILE__);
#endif

    }


}


void gpu_conv2DT_back(ConvolDescriptorT *D){


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
    float beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(hdnn[device], &alpha, D->wDesc, D->K->ptr,
                                             D->yDesc, D->D->ptr,
                                             D->convolution_descriptor, D->bwd_data_algorithm,
                                             shared_workspace[device], workspace_size[device],
                                             &beta, D->xDesc, D->ID->ptr),"cudnnConvolutionBackwardData",__FILE__);
#endif

}
