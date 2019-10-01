#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "tensor_cuda.h"
#include "tensor_kernels.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"



void gpu_set(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  set<<<dimGrid,dimBlock>>>(A->ptr,v,r,c);
  check_cuda(cudaDeviceSynchronize(),"set");

}


void gpu_copy_to_gpu(float *nptr,Tensor *A){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(A->ptr,nptr,A->size*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy_to_gpu");
}


void gpu_copy_from_gpu(Tensor *A,float *nptr){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(nptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy_to_gpu");
}


void gpu_copy_gpu(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);
  check_cuda(cudaMemcpy(B->ptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
}


void gpu_fill(Tensor *A,int aini,int aend,Tensor *B,int bini,int bend,int inc){
  int device=A->gpu_device;
  cudaSetDevice(device);

  int at=A->size/A->shape[0];
  int bt=B->size/B->shape[0];

  int t=1;
  for(int i=2;i<B->ndim;i++)
    t*=B->shape[i];

  int tot=B->shape[0]*(bend-1)*B->shape[1]*t;
  int r,c;

  while (aend-aini>0) {

      if ((aend-aini)>MAX_TPB) r=MAX_TPB;
      else r=(aend-aini);
      c=t;

      dim3 dimGrid(A->shape[0],c);
      dim3 dimBlock(r);

      fill<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,t,aini,at,bini,bt,tot,inc);
      aini+=MAX_TPB;
      bini+=MAX_TPB;

  }

    //check_cuda(cudaDeviceSynchronize(),"fill");

}



void gpu_mask(Tensor *A,float v) {

  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A)

  mask<<<dimGrid,dimBlock>>>(A->ptr,v,A->shape[0],c);
  check_cuda(cudaDeviceSynchronize(),"mask");

}
