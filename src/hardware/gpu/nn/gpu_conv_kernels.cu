/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <string.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda.h>

#include "eddl/hardware/gpu/nn/gpu_tensor_nn_kernels.h"
#include "eddl/hardware/gpu/gpu_kernels.h"


__global__ void  gpu_traspose_batch_depth(float *ptrB, float *ptr, int b,int z,int r, int c)
{
  long int ops=b*z*r*c;
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  if (thread_id_x < ops) {
    int bo=thread_id_x/(z*r*c);
    int zom=thread_id_x%(z*r*c);
    int zo=zom/(r*c);
    int rom=zom%(r*c);
    int ro=rom/c;
    int co=rom%c;

    int pos=(zo*(b*r*c))+(bo*(r*c))+(ro*c)+co;

    ptr[thread_id_x]=ptrB[pos];

  }

}

__global__ void  gpu_addbias_k(float *O, int batch, int r,int c,int nk,float *bias,int offset)
{
  int size=nk*r*c;
  int thread_id_x=threadIdx.x;

  int p=blockIdx.x*size+(thread_id_x+offset)*r*c;
  for (int i = 0; i < r*c; i++)
     O[p+i]+=bias[thread_id_x+offset];

}

__global__ void  gpu_deltabias_k(float *D, int batch, int r,int c,int nk,float *bias, int offset)
{
  int size=nk*r*c;
  int thread_id_x=threadIdx.x;

  int p=blockIdx.x*size+(thread_id_x+offset)*r*c;
  for (int i = 0; i < r*c; i++)
    atomicAdd(&(bias[thread_id_x+offset]),D[p+i]);

}


__global__ void gpu_im2col_k(float* I, float *ptrI,int batch,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int padrt,int padrb,int padcl,int padcr,int col2im)
{
  long int ops=batch*orows*ocols*kr*kc*idepth;
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  if (thread_id_x < ops) {
    int iz,ix,iy;

    int ksize=kr*kc*idepth;

    int im=thread_id_x/(ksize*orows*ocols);
    int ioffset=im*irows*icols*idepth;


    int tx=thread_id_x%(ksize*orows*ocols);


    int r=tx/ksize;
    int c=tx%ksize;

    int oy=r/ocols;
    int ox=r%ocols;

    ix=(ox*sc)-padcl;
    iy=(oy*sr)-padrt;
    iz=c/(kr*kc);

    c=c%(kr*kc);

    iy+=c/kc;
    ix+=c%kc;

    if ((ix>=0)&&(ix<icols)&&(iy>=0)&&(iy<irows)) {
      int p=iz*(irows*icols)+(iy*icols)+ix;
      if (col2im)
        atomicAdd(&(I[p+ioffset]),ptrI[thread_id_x]);
      else
	ptrI[thread_id_x]=I[p+ioffset];
    }
    else
      if (!col2im)
        ptrI[thread_id_x]=0;

  }

}

__global__ void gpu_im2col_k_low(float* I, int b, float *ptrI,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int padrt,int padrb,int padcl,int padcr,int col2im)
{
  long int ops=orows*ocols*kr*kc*idepth;
  long int thread_id_x = threadIdx.x+blockIdx.x*blockDim.x;


  if (thread_id_x < ops) {
    int iz,ix,iy;

    int ksize=kr*kc*idepth;

    int im=b;
    int ioffset=im*irows*icols*idepth;


    int tx=thread_id_x%(ksize*orows*ocols);


    int r=tx/ksize;
    int c=tx%ksize;

    int oy=r/ocols;
    int ox=r%ocols;

    ix=(ox*sc)-padcl;
    iy=(oy*sr)-padrt;
    iz=c/(kr*kc);

    c=c%(kr*kc);

    iy+=c/kc;
    ix+=c%kc;

    if ((ix>=0)&&(ix<icols)&&(iy>=0)&&(iy<irows)) {
      int p=iz*(irows*icols)+(iy*icols)+ix;
      if (col2im)
        atomicAdd(&(I[p+ioffset]),ptrI[thread_id_x]);
      else
      	ptrI[thread_id_x]=I[p+ioffset];
    }
    else
      if (!col2im)
        ptrI[thread_id_x]=0;

  }

}
