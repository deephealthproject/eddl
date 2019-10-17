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
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include "gpu_nn.h"
#include "gpu_nn_kernels.h"

#include "../gpu_hw.h"
#include "../gpu_tensor.h"
#include "../gpu_kernels.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


void gpu_relu(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  relu<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD) {
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


void gpu_sigmoid(Tensor *A,Tensor *B){
  int device=A->gpu_device;
  cudaSetDevice(device);

  setDims(A);

  sigmoid<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}

void gpu_d_sigmoid(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_sigmoid<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}


class exp_max_functor {

    float max;

    public:

        exp_max_functor(float m_) { max = m_; }

        __host__ __device__ float operator()(float x) const
        {
            return expf(x-max);
        }
};

class inv_sum_functor {

    float sum;

    public:

        inv_sum_functor(float s_) { sum = s_; }

        __host__ __device__ float operator()(float x) const
        {
            return x/sum;
        }
};

void gpu_softmax(Tensor *A,Tensor *B){

  int device=A->gpu_device;
  cudaSetDevice(device);

  int r,c;
  r=A->shape[0];
  c=A->shape[1];

    /*
    check_cuda(cudaMemcpy(B->ptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
    for(int i=0;i<r;i++,B->ptr) {
      float *ptr=B->ptr+(i*c);
      thrust::device_ptr<float> dptr = thrust::device_pointer_cast(ptr);

      float max=*(thrust::max_element(dptr, dptr + c));
      thrust::transform(dptr, dptr + c, dptr,exp_max_functor(max));
      float sum=thrust::reduce(dptr,dptr+c);
      thrust::transform(dptr, dptr + c, dptr,inv_sum_functor(sum));
    }
    */

  dim3 dimGrid(1);
  dim3 dimBlock(MAX_TPB);

  int i;
  for(i=0;i<r/MAX_TPB;i++) {
    float *aptr=A->ptr+(i*MAX_TPB*c);
    float *bptr=B->ptr+(i*MAX_TPB*c);
    int size=MAX_TPB*c;

    float* aux=gpu_create_tensor(device,size);
    softmax<<<dimGrid,dimBlock>>>(aptr,bptr,aux,c,size);
    check_cuda(cudaDeviceSynchronize(),"gpu_relu");
    gpu_delete_tensor(device,aux);
  }

  if (r%MAX_TPB) {
    dim3 dimGridm(1);
    dim3 dimBlockm(r%MAX_TPB);
    float *aptr=A->ptr+(i*MAX_TPB*c);
    float *bptr=B->ptr+(i*MAX_TPB*c);
    int size=(r%MAX_TPB)*c;

    float* aux=gpu_create_tensor(device,size);
    softmax<<<dimGridm,dimBlockm>>>(aptr,bptr,aux,c,size);
    check_cuda(cudaDeviceSynchronize(),"gpu_relu");
    gpu_delete_tensor(device,aux);
  }
}



void gpu_d_softmax(Tensor *D,Tensor *I,Tensor *PD){
  int device=D->gpu_device;
  cudaSetDevice(device);

  setDims(D)

  d_relu<<<dimGrid,dimBlock>>>(D->ptr,I->ptr,PD->ptr,D->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_relu");
}
