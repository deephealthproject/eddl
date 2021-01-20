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

int allocate_workspace(size_t size){
    if (size <= workspace_size){
        std::cout<<size <<" is smaller than "<<workspace_size<<" so return"<<std::endl;
        return 0;
    }
    else {
        std::cout<<size <<" is bigger than "<<workspace_size<<" so update"<<std::endl;
        workspace_size = size;
        cudaFree(shared_workspace);
        return cudaMalloc((void **) &shared_workspace, size);
    }
}


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
  std::cout<<"starting convolution... init?"<< D->cudnn_env_init <<std::endl;
  if (D->cudnn_env_init < 0){
      D->cudnn_env_init = 1;
      int requestedAlgoCount;
      //check_cudnn(cudnnGetConvolutionForwardAlgorithmMaxCount(D->cudnn_handle, &requestedAlgoCount));
      cudnnStatus_t bbb = cudnnGetConvolutionForwardAlgorithmMaxCount(
              D->cudnn_handle, &requestedAlgoCount);
  if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error 1 "<< cudnnGetErrorString(bbb) <<std::endl;
      int returnedAlgoCount;
      cudnnConvolutionFwdAlgoPerf_t * perfResults = new cudnnConvolutionFwdAlgoPerf_t [requestedAlgoCount];
      //check_cudnn(cudnnFindConvolutionForwardAlgorithm( D->cudnn_handle, D->xDesc, D->wDesc, D->convolution_descriptor, D->yDesc,
      //            requestedAlgoCount, &returnedAlgoCount, perfResults));
      bbb = cudnnFindConvolutionForwardAlgorithm( D->cudnn_handle, D->xDesc, D->wDesc, D->convolution_descriptor, D->yDesc,
                  requestedAlgoCount, &returnedAlgoCount, perfResults);
  if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error 2 "<< cudnnGetErrorString(bbb) <<std::endl;
      int aux_alg = 0;
      size_t size;
      do{
          cout<<"Miro alg"<<aux_alg<<endl;
          D->fwd_algorithm = perfResults[aux_alg].algo;
          cout<<D->cudnn_handle<<endl;

    cudnnDataType_t         dataType;
    int                     n;
    int                     c;
    int                     h;
    int                     w;
    int                     nStride;
    int                     cStride;
    int                     hStride;
    int                     wStride;
    cudnnTensorFormat_t        format;
    bbb = cudnnGetTensor4dDescriptor(D->xDesc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride);
  if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"get xDesc "<< cudnnGetErrorString(bbb) <<std::endl;
    std::cout <<"xDesc: "<<dataType<<", "<< n<<", " <<c<<", " <<h<< ", "<<w << ", " <<nStride <<", " <<cStride<<", "<<hStride<<", " <<wStride<<std::endl;
    bbb = cudnnGetFilter4dDescriptor(D->wDesc, &dataType, &format, &n, &c, &h, &w);
    if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"get wDesc "<< cudnnGetErrorString(bbb) <<std::endl;
    std::cout <<"wDesc: "<<dataType<<", "<< n<<", " <<c<<", " <<h<< ", "<<w <<std::endl;
    bbb = cudnnGetTensor4dDescriptor(D->yDesc, &dataType, &n, &c, &h, &w, &nStride, &cStride, &hStride, &wStride);
  if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"get yDesc "<< cudnnGetErrorString(bbb) <<std::endl;
    std::cout <<"yDesc: "<<dataType<<", "<< n<<", " <<c<<", " <<h<< ", "<<w << ", "<<nStride <<", " <<cStride<<", " <<hStride<<", " <<wStride<<std::endl;
          //check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(D->cudnn_handle,D->xDesc, D->wDesc,
          //                                                    D->convolution_descriptor,  D->yDesc,
          //                                                    D->fwd_algorithm, &size));
          bbb =cudnnGetConvolutionForwardWorkspaceSize(D->cudnn_handle,D->xDesc, D->wDesc,
                                                              D->convolution_descriptor,  D->yDesc,
                                                              D->fwd_algorithm, &size);
  if(bbb != CUDNN_STATUS_SUCCESS) std::cout<<"Error 3 "<< cudnnGetErrorString(bbb) <<std::endl;
          aux_alg++;
      }
      while(allocate_workspace(size));
  std::cout<<" convolution env created... init?"<< D->cudnn_env_init <<std::endl;
  }
  cudnnStatus_t aaa = cudnnConvolutionForward( D->cudnn_handle, &alpha, D->xDesc, D->I->ptr,
                                       D->wDesc, D->K->ptr,
                                       D->convolution_descriptor, D->fwd_algorithm,
                                       shared_workspace, workspace_size,
                                       &beta, D->yDesc, D->O->ptr);
  if(aaa != CUDNN_STATUS_SUCCESS) std::cout<<"Error en convolucion "<< cudnnGetErrorString(aaa) <<std::endl;
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
