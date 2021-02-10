/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <string>
#include <stdexcept>
#include <iostream>

#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_kernels.h"

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
#ifdef cCUDNN
cudnnStatus_t dstatus;
cudnnHandle_t hdnn[64];
#endif

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
            std::string text = "unknown curand error: " + std::to_string(error) + " | (_curandGetErrorEnum)";
            throw std::invalid_argument(text);
    }

}

void check_cublas(cublasStatus_t status, const char *f)
{
    if ( status!=  CUBLAS_STATUS_SUCCESS)
    {
        std::string text = "error in cublas execution in " + std::string(f) + " | (check_cublas)";
        throw std::runtime_error(text);
    }
}

void check_curand(curandStatus_t status, const char *f)
{
    if ( status!=  CURAND_STATUS_SUCCESS)
    {
        std::string text = "error in curand execution in " + std::string(_curandGetErrorEnum(status)) + " | (check_curand)";
        throw std::runtime_error(text);
    }
}


void check_cuda(cudaError_t err,const char *msg)
{
    if(err!=cudaSuccess)
    {
        std::string error_type = cudaGetErrorString(err);
        std::string text = "[CUDA ERROR]: " + error_type + " ("+ std::to_string(err) + ") raised in " + std::string(msg) + " | (check_cuda)";
        throw std::runtime_error(text);
    }

}

#ifdef cCUDNN

void check_cudnn(cudnnStatus_t status, const char *msg, const char *file)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        std::string error_type = cudnnGetErrorString(status);
        std::string text = "[CUDNN ERROR]: " + error_type + " ("+ std::to_string(status) + ") raised in " + std::string(msg) + " at " + std::string(file) + " file | (check_cudnn)";
        throw std::runtime_error(text);
    }
}

#endif
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
        std::string text = "GPU " + std::to_string(device) + " not available. Number of available GPUs is " + std::to_string(nDevices) + ". Further information running nvidia-smi  | (gpu_init)";
        throw std::runtime_error(text);
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
        std::string text = "problem in cublas create (gpu_init)";
        throw std::runtime_error(text);
    }
    fprintf(stderr,"CuBlas initialized on GPU device %d, %s\n",device,prop.name);

    bstatus = cublasSetAtomicsMode(hcublas[device],CUBLAS_ATOMICS_NOT_ALLOWED);
    if ( bstatus!=  CUBLAS_STATUS_SUCCESS)
    {
        std::string text = "problem in cublas execution getting: NOT IMPLEMENTED |  (gpu_init)";
        throw std::runtime_error(text);
    }

    // CURAND
    rstatus=curandCreateGenerator(&(random_generator[device]),CURAND_RNG_PSEUDO_MRG32K3A);
    if (rstatus != CURAND_STATUS_SUCCESS)
    {
        std::string text = "error creating random numbers on gpu | (gpu_init)";
        throw std::runtime_error(text);
    }
    rstatus=curandSetPseudoRandomGeneratorSeed(random_generator[device],1234);

    if (rstatus != CURAND_STATUS_SUCCESS) {
        std::string text = "error setting the seed for program | (gpu_init)";
        throw std::runtime_error(text);
    }
    fprintf(stderr,"CuRand initialized on GPU device %d, %s\n",device,prop.name);
#ifdef cCUDNN
    // CUDNN
    dstatus=cudnnCreate(&hdnn[device]);
    if (dstatus != CUDNN_STATUS_SUCCESS) {
        std::string text = "problem in cudnn create (gpu_init)";
        throw std::runtime_error(text);
    }

    fprintf(stderr,"CuDNN initialized on GPU device %d, %s\n",device,prop.name);

#endif

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
