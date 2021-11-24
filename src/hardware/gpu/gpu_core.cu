/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>


#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"
#include "eddl/hardware/gpu/gpu_hw.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"


int* get_block_dim(int N, int blockSize){
    int* res = new int[2];
    int blocks = (N + blockSize - 1) / blockSize;
    if (N<blockSize) { blockSize = N; }

    res[0] = blocks;
    res[1] = blockSize;
    return res;
}

void copy_cpu2gpu(float *cpu_addresses, float* gpu_addresses, int size, bool delete_cpu){
    check_cuda(cudaMalloc((void**)&(gpu_addresses), size), "create address mapping");
    check_cuda(cudaDeviceSynchronize(), "create");


    check_cuda(cudaMemcpy(gpu_addresses, cpu_addresses, size, cudaMemcpyHostToDevice), "copy address mapping");
    check_cuda(cudaDeviceSynchronize(), "copy");

    // Free CPU pointer?
    if (delete_cpu) { delete[] cpu_addresses; }
}

void gpu_copy_to_gpu(float *nptr,Tensor *A){
    int device=A->gpu_device;
    cudaSetDevice(device);
    check_cuda(cudaMemcpy(A->ptr,nptr,A->size*sizeof(float),cudaMemcpyHostToDevice),"gpu_copy_to_gpu");
}


void gpu_copy_from_gpu(Tensor *A, float *nptr){
    int device=A->gpu_device;
    cudaSetDevice(device);
    check_cuda(cudaMemcpy(nptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToHost),"gpu_copy_to_gpu");
}


void gpu_copy_gpu(Tensor *A,Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);
    check_cuda(cudaMemcpy(B->ptr,A->ptr,A->size*sizeof(float),cudaMemcpyDeviceToDevice),"gpu_copy_gpu");
}


void cpu2gpu(float *dst, const float *src, unsigned long int size, int gpu_device){
    cudaSetDevice(gpu_device);
    check_cuda(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyHostToDevice),"cpu2gpu");
}

void gpu2cpu(float *dst, const float *src, unsigned long int size, int gpu_device){
    cudaSetDevice(gpu_device);
    check_cuda(cudaMemcpy(dst, src, size*sizeof(float), cudaMemcpyDeviceToHost),"gpu2cpu");
}

float* get_gpu_fmem(unsigned long int size, int gpu_device){
    float* ptr;
    cudaSetDevice(gpu_device);
    check_cuda(cudaMalloc((void**)&ptr,size*sizeof(float)),"get_gpu_fmem");
    return ptr;
}

void free_gpu_ptr(float *ptr, int gpu_device){
    cudaSetDevice(gpu_device);
    check_cuda(cudaFree(ptr),"free_gpu_ptr");
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

    setDims(A);

    mask<<<dimGrid,dimBlock>>>(A->ptr,v,A->size);
    check_cuda(cudaDeviceSynchronize(),"mask");

}


void gpu_fill_(Tensor *A, float v) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    fill_<<<dimGrid,dimBlock>>>(A->ptr,v,A->size);
    check_cuda(cudaDeviceSynchronize(),"set");
}

void gpu_select(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, bool mask_zeros)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  int *ind;
  cudaMalloc((void **) &ind, sind.size() * sizeof(int));
  cudaMemcpy(ind, &sind[0], sind.size() * sizeof(int), cudaMemcpyHostToDevice);



  int size=sind.size()*(B->shape[1]);

  int grid,block;
  if (size>=1024) {
    grid=size/1024;
    block=1024;
    if (size%1024) grid++;
  }
  else {
    grid=1;
    block=size;
  }

  dim3 dimGrid(grid);
  dim3 dimBlock(block);

  select_rows<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->shape[1], size, ind, ini, mask_zeros);
  check_cuda(cudaDeviceSynchronize(), "gpu_select");

  cudaFree(ind);

}

void gpu_deselect(Tensor *A, Tensor *B, vector<int> sind, int ini, int end, int inc, bool mask_zeros)
{
  int device=A->gpu_device;
  cudaSetDevice(device);

  int *ind;
  cudaMalloc((void **) &ind, sind.size() * sizeof(int));
  cudaMemcpy(ind, &sind[0], sind.size() * sizeof(int), cudaMemcpyHostToDevice);

  int size=sind.size()*(B->shape[1]);

  int grid,block;
  if (size>=1024) {
    grid=size/1024;
    block=1024;
    if (size%1024) grid++;
  }
  else {
    grid=1;
    block=size;
  }

  dim3 dimGrid(grid);
  dim3 dimBlock(block);

  deselect_rows<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->shape[1], size, ind, ini, inc,mask_zeros);
  check_cuda(cudaDeviceSynchronize(), "gpu_select");

  cudaFree(ind);

}


void gpu_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

     if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }


    setDims(B);  // B is the small
    select<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "select");
}

void gpu_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, A->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), A->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, A->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }


    setDims(A);  // A is the small
    select_back<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "select_back");
}


void gpu_set_select(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    set_select<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "set_select");
}


void gpu_set_select_back(Tensor *A, Tensor *B, SelDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    set_select_back<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "set_select_back");
}

void gpu_gather(Tensor *A, Tensor *B, GatherDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    gpu_gather<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "gpu_gather");
}


void gpu_expand(Tensor *A, Tensor *B, ExpandDescriptor *sd){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Copy indices from host to device
    if(sd->gpu_addresses == nullptr){
        // copy_cpu2gpu(sd->cpu_addresses, sd->gpu_addresses, B->size*sizeof(int), true);

        check_cuda(cudaMalloc((void**)&(sd->gpu_addresses), B->size*sizeof(int)), "create address mapping");
        check_cuda(cudaDeviceSynchronize(), "create");

        check_cuda(cudaMemcpy(sd->gpu_addresses, sd->cpu_addresses, B->size*sizeof(int), cudaMemcpyHostToDevice), "copy address mapping");
        check_cuda(cudaDeviceSynchronize(), "copy");
    }

    setDims(B);  // B is the small
    gpu_expand<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, B->size, sd->gpu_addresses);
    check_cuda(cudaDeviceSynchronize(), "gpu_expand");
}


void  gpu_repeat(Tensor* A, Tensor *B, vector<unsigned int> repeats, unsigned int axis){
    int device=A->gpu_device;
    cudaSetDevice(device);

    unsigned int *gpu_vrepeats;
    cudaMalloc((void **) &gpu_vrepeats, repeats.size() * sizeof(unsigned int));
    cudaMemcpy(gpu_vrepeats, &repeats[0], repeats.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int *gpu_A_shape;
    cudaMalloc((void **) &gpu_A_shape, A->ndim * sizeof(unsigned int));
    cudaMemcpy(gpu_A_shape, A->shape.data(), A->ndim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int *gpu_B_shape;
    cudaMalloc((void **) &gpu_B_shape, B->ndim * sizeof(unsigned int));
    cudaMemcpy(gpu_B_shape, B->shape.data(), B->ndim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int *gpu_A_strides;
    cudaMalloc((void **) &gpu_A_strides, A->ndim * sizeof(unsigned int));
    cudaMemcpy(gpu_A_strides, A->stride.data(), A->ndim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned int *gpu_B_strides;
    cudaMalloc((void **) &gpu_B_strides, B->ndim * sizeof(unsigned int));
    cudaMemcpy(gpu_B_strides, B->stride.data(), B->ndim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    setDims(A);  // A is the small one. We should use B as is the big one, but this is a direct translation from the CPU code
    gpu_repeat<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, gpu_vrepeats, axis, A->size, B->size, 
                                     gpu_A_shape, gpu_B_shape, gpu_A_strides, gpu_B_strides, A->ndim, repeats.size());
    check_cuda(cudaDeviceSynchronize(), "gpu_repeat");
}

void gpu_repeat_batch(Tensor *A, Tensor *B){
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(B);  // B is the big one
    gpu_repeat_batch<<<dimGrid,dimBlock>>>(A->ptr, B->ptr, A->size, B->size);
    check_cuda(cudaDeviceSynchronize(), "gpu_repeat_batch");
}


void gpu_concat(Tensor *A, vector<Tensor*> t, unsigned int axis, bool derivative){
    int device=A->gpu_device;
    cudaSetDevice(device);

    // Walk through all the tensors to concat one axis (once)
    //unsigned int offset = 0;
    unsigned int size = 0;
    int steps = A->stride[axis] * A->shape[axis];  // Equivalent to A->stride[axis-1], but without the negative index problem

    // Walk through each tensor
    #pragma omp parallel for
    for (unsigned int i = 0; i < t.size(); i++) {
        int offset = i*size;
        size = t[i]->stride[axis] * t[i]->shape[axis];

        // Copy n bytes from src to dest
        float *dest = A->ptr + offset;
        float *src = t[i]->ptr;


        setDims(t[i]);
        concat<<<dimGrid,dimBlock>>>(dest, src, t[i]->size, size, steps, derivative);
        check_cuda(cudaDeviceSynchronize(),"gpu_concat");

    }
}



void gpu_sort(Tensor *A, Tensor *B, bool descending, bool stable){
    auto order_desc = thrust::greater<float>();
    auto order_asc = thrust::less<float>();

    // Copy data from A to B
    thrust::device_ptr<float> A_d(A->ptr);
    thrust::device_ptr<float> B_d(B->ptr);
    thrust::copy(A_d, A_d+A->size, B_d);

    // Sort data
    if(stable) {
        if (descending) { thrust::stable_sort(B_d, B_d + B->size, order_desc); }
        else { thrust::stable_sort(B_d, B_d + B->size, order_asc); }
    } else{
        if (descending) { thrust::sort(B_d, B_d+B->size, order_desc); }
        else { thrust::sort(B_d, B_d+B->size, order_asc); }
    }
}

void gpu_argsort(Tensor *A, Tensor *B, bool descending, bool stable) {
    auto order_desc = thrust::greater<float>();
    auto order_asc = thrust::less<float>();

    // Copy data from A to B
    thrust::device_ptr<float> keys(A->ptr);  // add this line before the sort line
    thrust::device_ptr<float> indices(B->ptr);  // add this line before the sort line
    
    // Fill B with indices
    thrust::sequence(indices, indices+B->size, 0);

    // Sort data
    if(stable){
        if (descending) { thrust::stable_sort_by_key(keys, keys+B->size, indices, order_desc); }
        else { thrust::stable_sort_by_key(keys, keys+B->size, indices, order_asc); }
    }else{
        if (descending) { thrust::sort_by_key(keys, keys+B->size, indices, order_desc); }
        else { thrust::sort_by_key(keys, keys+B->size, indices, order_asc); }
    }
}
