/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_GPU_TENSOR_H
#define EDDL_GPU_TENSOR_H


#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#ifdef cCUDNN
#include <cudnn.h>
#endif
//
//#include <cstdio>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <cublas_v2.h>
//
//#include "eddl/hardware/gpu/gpu_tensor.h"
//#include "eddl/hardware/gpu/tensor_kernels.h"

// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(A) int setdim_r,setdim_c;setdim_r=(A->size/MAX_TPB);if (setdim_r==0) {setdim_r=1;setdim_c=A->size;}else {if (A->size%MAX_TPB) setdim_r++;setdim_c=MAX_TPB;}dim3 dimGrid(setdim_r);dim3 dimBlock(setdim_c);

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];

void check_cublas(cublasStatus_t status, const char *f);

void check_curand(curandStatus_t status, const char *f);
#ifdef cCUDNN
void check_cudnn(cudnnStatus_t status);
#endif

void check_cuda(cudaError_t err,const char *msg);
void gpu_set_device(int device);
void gpu_init(int device);

float* gpu_create_tensor(int dev,int size);
void gpu_delete_tensor(int dev,float* p);
void gpu_delete_tensor_int(int dev, int* p);

int gpu_devices();

#endif
