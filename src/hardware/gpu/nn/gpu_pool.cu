#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "gpu_nn.h"
#include "gpu_nn_kernels.h"

#include "../gpu_hw.h"
#include "../gpu_tensor.h"
#include "../gpu_kernels.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


void gpu_mpool2D(PoolDescriptor *D){
    int device=D->I->gpu_device;
    cudaSetDevice(device);

    setDims(D->O);
    maxpool2d<<<dimGrid,dimBlock>>>(D->I->ptr, D->I->shape[0],D->ir,D->ic,D->iz,D->kr,D->kc,D->O->ptr,D->r,D->c,D->z, D->sr,D->sc,D->padr, D->padc, D->indX->ptr, D->indY->ptr);

    check_cuda(cudaDeviceSynchronize(),"gpu_mpool");
}


void gpu_mpool2D_back(PoolDescriptor *D){
    int device=D->I->gpu_device;
    cudaSetDevice(device);

    setDims(D->D)
    maxpool2d_back<<<dimGrid,dimBlock>>>(D->I->ptr, D->I->shape[0],D->ir,D->ic,D->iz,D->kr,D->kc, D->sr,D->sc,D->padr, D->padc, D->indX->ptr, D->indY->ptr, D->D->ptr, D->ID->ptr);

    check_cuda(cudaDeviceSynchronize(),"gpu_mpool_back");
}

