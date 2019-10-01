#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../tensor_cuda.h"
#include "../tensor_kernels.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"

void gpu_im2col(ConvolDescriptor *D, int col2im){
  int device=D->I->gpu_device;
  cudaSetDevice(device);

  setDims(D->gpuI)
  dimGrid.x*=D->I->shape[0];

  if (col2im)
    gpu_im2col_k<<<dimGrid,dimBlock>>>(D->ID->ptr, D->gpuI->ptr,D->I->shape[0],D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padr,1);
  else
    gpu_im2col_k<<<dimGrid,dimBlock>>>(D->I->ptr, D->gpuI->ptr,D->I->shape[0],D->ir,D->ic,D->iz,D->K->ptr,D->nk,D->kr,D->kc,D->O->ptr,D->r,D->c,D->sr,D->sc,D->padr,0);

  check_cuda(cudaDeviceSynchronize(),"gpu_im2col");

}


void gpu_conv2D(ConvolDescriptor *D) {

  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;

  //gpuIB=new Tensor(vector<int>{A->shape[0]*r*c,kc*kr*kz}, I->device);

  int isize=D->kz*D->kr*D->kc*D->r*D->c;

  D->gpuK->ptr=D->K->ptr;
  D->gpuO->ptr=D->O->ptr;
  D->gpuI->ptr=D->gpuIB->ptr;

  gpu_im2col(D,0);

  for(int b=0;b<D->I->shape[0];b++,D->gpuO->ptr+=osize,D->gpuI->ptr+=isize)
    gpu_mult2D(D->gpuK,0,D->gpuI,1,D->gpuO,0);


  gpu_addbias_k<<<D->O->shape[0],D->bias->shape[0]>>>(D->O->ptr, D->O->shape[0], D->r,D->c,D->nk,D->bias->ptr);

  check_cuda(cudaDeviceSynchronize(),"gpu_addbias");

}


void gpu_conv2D_grad(ConvolDescriptor *D){

  int device=D->I->gpu_device;

  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;
  int isize=D->kz*D->kr*D->kc*D->r*D->c;

  D->gpugK->ptr=D->gK->ptr;
  D->gpuD->ptr=D->D->ptr;
  D->gpuI->ptr=D->gpuIB->ptr;

  for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize,D->gpuI->ptr+=isize){
    gpu_mult2D(D->gpuD,0,D->gpuI,0,D->gpugK,1);
    }


  gpu_deltabias_k<<<D->D->shape[0],D->bias->shape[0]>>>(D->D->ptr, D->D->shape[0], D->r,D->c,D->nk,D->gbias->ptr);

  check_cuda(cudaDeviceSynchronize(),"gpu_deltabias");

}


void gpu_conv2D_back(ConvolDescriptor *D){


  int device=D->I->gpu_device;
  cudaSetDevice(device);

  int osize=D->z*D->r*D->c;
  int isize=D->kz*D->kr*D->kc*D->r*D->c;


  D->gpuK->ptr=D->K->ptr;
  D->gpuD->ptr=D->D->ptr;
  D->gpuI->ptr=D->gpuIB->ptr;

  for(int b=0;b<D->I->shape[0];b++,D->gpuD->ptr+=osize,D->gpuI->ptr+=isize) {
      gpu_mult2D(D->gpuD, 1, D->gpuK, 0, D->gpuI, 0);
  }

  D->gpuI->ptr=D->gpuIB->ptr;
  gpu_im2col(D,1);
}

