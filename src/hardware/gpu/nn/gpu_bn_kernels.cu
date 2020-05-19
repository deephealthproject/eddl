/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/nn/gpu_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"


__global__ void bn_permute_channels_last(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(bo*(r*c*z))+(ro*(c*z))+(co*z)+zo;
    dest[pos]=src[thread_id_x];
  }
}

__global__ void bn_permute_channels_first(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(bo*(r*c*z))+(ro*(c*z))+(co*z)+zo;
    dest[thread_id_x]=src[pos];
  }
}


__global__ void bn_permute_batch_last(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(zo*(r*c*b))+(ro*(c*b))+(co*b)+bo;
    dest[pos]=src[thread_id_x];
  }
}

__global__ void bn_permute_batch_first(float *src, float *dest,int b,int z,int r,int c,long int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x < size) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(zo*(r*c*b))+(ro*(c*b))+(co*b)+bo;
    dest[thread_id_x]=src[pos];
  }
}
