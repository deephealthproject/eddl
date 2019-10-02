#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_nn.h"
#include "../gpu_hw.h"
#include "../gpu_tensor.h"
#include "../gpu_kernels.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


void gpu_cent(Tensor *A,Tensor *B,Tensor *C){

  int device=A->gpu_device;
  cudaSetDevice(device);
  setDims(A);

  cent<<<dimGrid,dimBlock>>>(A->ptr,B->ptr,C->ptr,A->size);
  check_cuda(cudaDeviceSynchronize(),"gpu_cent");
}
