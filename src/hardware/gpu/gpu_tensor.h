/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es), (jmaronasm@gmail.com)
* All rights reserved
*/


#ifndef _TENSOR_CUDA_
#define _TENSOR_CUDA_

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

//
//#include <cstdio>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
//#include <cublas_v2.h>
//
//#include "hardware/gpu/gpu_tensor.h"
//#include "hardware/gpu/tensor_kernels.h"



void check_cublas(cublasStatus_t status, const char *f);

void check_curand(curandStatus_t status, const char *f);

void check_cuda(cudaError_t err,const char *msg);
void gpu_set_device(int device);
void gpu_init(int device);

float* gpu_create_tensor(int dev,int size);
void gpu_delete_tensor(int dev,float* p);
void gpu_delete_tensor_int(int dev, int* p);

int gpu_devices();

#endif
