/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Juan Maroñas: jmaronas@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>

#include "gpu_kernels.h"

__global__ void reduction_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int max)
{
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  int j;
  float sum=0;
  float v,val;

  int i;

  int p=max*blockIdx.x;


  for(j=0;j<max && ind[p]!=-1;j++,p++) {
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

  p=max*blockIdx.x;
  // set in Output
  if (m<2) { // mean or sum
      if (m==0) sum/=d;
      if (keepdims) {
        for(j=0;j<max&& ind[p]!=-1;j++,p++)
            O[ind[p]]=sum;
      }
      else O[thread_id_x]=sum;
  }
  else { // max or min
      if (keepdims) {
        for(j=0;j<max && ind[p]!=-1;j++,p++) {
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



__global__ void reduction_back_kernel(float *I,float *O,float *S,int m, int keepdims,int d,int *ind,int max)
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
      p=max*blockIdx.x;
      if(keepdims) {
        for(j=0;j<max && ind[p]!=-1;j++,p++)
          val+=I[ind[p]];
      }
      else val=I[thread_id_x];
      if (m==0) val/=d;

      p=max*blockIdx.x;
      for(j=0;j<max && ind[p]!=-1;j++,p++)
        O[ind[p]]+=val;
    }
}
