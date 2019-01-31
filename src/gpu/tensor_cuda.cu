// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// The MIT License (MIT)
//
// Copyright (c) 2019 Roberto Paredes Palacios, <rparedes@dsic.upv.es>

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include "tensor_cuda.h"

cublasHandle_t hcublas[64];
curandGenerator_t random_generator[64];
cublasStatus_t bstatus;
curandStatus_t rstatus;



void check_cuda(cudaError_t err,char *msg)
{
  if(err!=cudaSuccess)
  {
     fprintf(stderr,"Cuda Error: %s\n",msg);
     exit(0);
  }

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
    fprintf(stderr,".");
  }
  fprintf(stderr,"\n");

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

void gpu_set_device(int device)
{
   cudaSetDevice(device);
}


///////////////

float* gpu_create_tensor(int size)
{
  float* devicePointer;
  check_cuda(cudaMalloc((void**)&devicePointer,size*sizeof(float)),"create_tensor");
  return devicePointer;
}
void gpu_delete_tensor(float* p)
{
  check_cuda(cudaFree(p),"delete_tensor");
}
