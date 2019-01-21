#include <stdio.h>
#include "tensor_cuda.h"


void check_cuda(cudaError_t err,char *msg)
{
  if(err!=cudaSuccess)
  {
     fprintf(stderr,"Cuda Error: %s\n",msg);
     exit(0);
  }

}

float* create_tensor(int size)
{
  float* devicePointer;
  check_cuda(cudaMalloc((void**)&devicePointer,size*sizeof(float)),"create_tensor");
  return devicePointer;
}


void delete_tensor(float* p)
{
  check_cuda(cudaFree(p),"delete_tensor");
}
