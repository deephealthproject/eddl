
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Juan Maroñas: jmaronas@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#ifndef _TENSOR_CUDA_
#define _TENSOR_CUDA_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "tensor_cuda.h"
#include "tensor_kernels.h"


void check_cublas(cublasStatus_t status, const char *f);

void check_curand(curandStatus_t status, const char *f);

void check_cuda(cudaError_t err,const char *msg);
void gpu_set_device(int device);
void gpu_init(int device);

float* gpu_create_tensor(int dev,int size);
void gpu_delete_tensor(int dev,float* p);

int gpu_devices();

#endif
