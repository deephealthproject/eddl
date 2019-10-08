#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_tensor.h"
#include "gpu_kernels.h"
#include "gpu_hw.h"

#include "../../tensor/tensor.h"
#include "../../descriptors/descriptors.h"

void gpu_range(Tensor *A, float min, float step, int size) {
    int device=A->gpu_device;
    cudaSetDevice(device);

    setDims(A);

    //range<<<dimGrid,dimBlock>>>(A->ptr, r, c, min, step, size);
    check_cuda(cudaDeviceSynchronize(),"range");
}
