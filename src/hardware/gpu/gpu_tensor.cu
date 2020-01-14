/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es), (jmaronasm@gmail.com)
* All rights reserved
*/


#include <stdio.h>
#include "gpu_tensor.h"
#include "gpu_kernels.h"

// CUDA, NVIDIA compute capabilities:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
// -----------------------------------------------------------------
//                      GRID
// Maximum dimensionality of grid of thread blocks:	3
// Maximum x-dimension of a grid of thread blocks	(2^31)-1
// Maximum y- or z-dimension of a grid of thread blocks: 65535
//                   THREAD BLOCK
// Maximum dimensionality of thread block:	3
// Maximum x- or y-dimension of a block:	1024
// Maximum z-dimension of a block:	64
//
// Maximum number of threads per block:	1024
// -----------------------------------------------------------------

cublasHandle_t hcublas[64];
curandGenerator_t random_generator[64];
cublasStatus_t bstatus;
curandStatus_t rstatus;

static const char *_curandGetErrorEnum(curandStatus_t error){
    switch (error)
    {
        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";


        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";


        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";


        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";


        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";


        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        default:
            fprintf(stderr,"Not all curand errors here %d\n",error);
            exit(-1);
    }

}

void check_cublas(cublasStatus_t status, const char *f)
{
    if ( status!=  CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"Error in cublas execution in %s\n",f);
        exit(1);
    }
}

void check_curand(curandStatus_t status, const char *f)
{
    if ( status!=  CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr,"Error in curand execution in %s\n",_curandGetErrorEnum(status));
        exit(1);
    }
}


void check_cuda(cudaError_t err,const char *msg)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"Cuda Error %d in %s\n",err,msg);
        exit(0);
    }

}


void gpu_set_device(int device)
{
    cudaSetDevice(device);
}


void gpu_init(int device)
{

    int nDevices;
    cudaGetDeviceCount(&nDevices);

    if (device>nDevices)
    {
        fprintf(stderr,"Error. GPU %d not available. Number of available GPU is %d. Further information running nvidia-smi\n",device,nDevices);
        exit(-1);
    }

    fprintf(stderr,"Selecting GPU device %d\n",device);
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);

    fprintf(stderr,"EDDLL is running on GPU device %d, %s\n",device,prop.name);


    /// CUBLAS
    bstatus=cublasCreate(&(hcublas[device]));
    // try to init cublas several times
    int i=0;
    while ((bstatus!=  CUBLAS_STATUS_SUCCESS)&&(i<10)) {
        bstatus=cublasCreate(&(hcublas[device]));
        i++;
        fprintf(stderr,".\n");
    }

    if ( bstatus!=  CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"Problem in cuBlas Create\n");
        exit(1);

    }
    fprintf(stderr,"CuBlas initialized on GPU device %d, %s\n",device,prop.name);

    bstatus = cublasSetAtomicsMode(hcublas[device],CUBLAS_ATOMICS_NOT_ALLOWED);
    if ( bstatus!=  CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"Problem in cuBlas execution getting: NOT IMPLEMENTED \n");
        exit(1);

    }

    // CURAND
    rstatus=curandCreateGenerator(&(random_generator[device]),CURAND_RNG_PSEUDO_MRG32K3A);
    if (rstatus != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr,"Error creating random numbers on gpu\n");
        exit(-1);
    }
    rstatus=curandSetPseudoRandomGeneratorSeed(random_generator[device],1234);

    if (rstatus != CURAND_STATUS_SUCCESS) {
        fprintf(stderr,"Error seeting the seed for program\n");
        exit(-1);
    }
    fprintf(stderr,"CuRand initialized on GPU device %d, %s\n",device,prop.name);



}


float* gpu_create_tensor(int dev,int size)
{
    float* devicePointer;
    cudaSetDevice(dev);
    check_cuda(cudaMalloc((void**)&devicePointer,size*sizeof(float)),"create_tensor");
    return devicePointer;
}


void gpu_delete_tensor(int dev, float* p)
{
    cudaSetDevice(dev);
    check_cuda(cudaFree(p),"delete_tensor");
}

void gpu_delete_tensor_int(int dev, int* p)
{
    cudaSetDevice(dev);
    check_cuda(cudaFree(p),"delete_tensor_int");
}

int gpu_devices()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    return nDevices;
}
