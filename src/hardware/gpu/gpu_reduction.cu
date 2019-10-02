#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "tensor_cuda.h"
#include "tensor_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"


// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(A) int r,c;r=(A->size/MAX_TPB);if (r==0) {r=1;c=A->size;}else {if (A->size%MAX_TPB) r++;c=MAX_TPB;}dim3 dimGrid(r);dim3 dimBlock(c);

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];