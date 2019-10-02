#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../tensor_cuda.h"
#include "../tensor_kernels.h"
#include "../gpu_hw.h"

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"


// MAX THREADS PER BLOCK
#define MAX_TPB 1024
#define setDims(A) int r,c;r=(A->size/MAX_TPB);if (r==0) {r=1;c=A->size;}else {if (A->size%MAX_TPB) r++;c=MAX_TPB;}dim3 dimGrid(r);dim3 dimBlock(c);

extern cublasHandle_t hcublas[64];
extern curandGenerator_t random_generator[64];


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

