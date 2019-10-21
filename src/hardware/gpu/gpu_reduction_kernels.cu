/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_kernels.h"
//dim3 dimGrid(RD->index.size());
//dim3 dimBlock(1);
__global__ void reduction_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  int j;
  float sum=0;
  float v,val;

  int i;

  int p=rs*blockIdx.x;


  for(j=0;j<rs;j++,p++) {
      v=I[ind[p]];
      if (m==2) {
          if (j==0) {val=v;i=p;}
          else if (v>val) {
              val=v;
              i=p;
          }
      }
      else if (m==3) {
        if (j==0) {val=v;i=p;}
        else if (v<val) {
            val=v;
            i=p;
        }
      }
      else sum+=v;
  }

  p=rs*blockIdx.x;
  // set in Output
  if (m<2) { // mean or sum
      if (m==0) sum/=d;
      if (keepdims) {
        for(j=0;j<rs;j++,p++)
            O[ind[p]]=sum;
      }
      else O[thread_id_x]=sum;
  }
  else { // rs or min
      if (keepdims) {
        for(j=0;j<rs;j++,p++) {
              O[ind[p]]=val;
              S[ind[p]]=i;
          }
      }
      else {
          O[thread_id_x]=val;
          S[thread_id_x]=i;
      }
  }

}


//dim3 dimGrid(RD->index.size());
//dim3 dimBlock(1);
__global__ void reduction_back_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int rs)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

    int j;
    float val=0;
    int p;

    // set in Delta
    if (m>=2) {
      int p=S[thread_id_x];
      O[p]+=I[thread_id_x];
    }
    else {
      p=rs*blockIdx.x;
      if(keepdims) {
        for(j=0;j<rs;j++,p++)
          val+=I[ind[p]];
      }
      else val=I[thread_id_x];
      if (m==0) val/=d;

      p=rs*blockIdx.x;
      for(j=0;j<rs;j++,p++)
        O[ind[p]]+=val;
    }
}



////////////////////
// FOR SUM and MEAN
// Faster in Conv
///////////////////

//dim3 dimGrid(red_size);
//dim3 dimBlock(RD->index.size());

__global__ void reduction_permute(float *I,float *O,int *ind,int size)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;

  if (thread_id_x<size)
    O[thread_id_x]=I[ind[thread_id_x]];
}

__global__ void reduction_kernel_keep(float *red, float *O, int *ind, int size, int rsize)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    if (thread_id_x<size*rsize) {
        O[ind[thread_id_x]]=red[thread_id_x/rsize];
    }
}

__global__ void reduction_kernel_keep_inc(float *red, float *O, int *ind, int size, int rsize)
{
    long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;
    if (thread_id_x<size*rsize) {
        O[ind[thread_id_x]]+=red[thread_id_x/rsize];
    }
}
