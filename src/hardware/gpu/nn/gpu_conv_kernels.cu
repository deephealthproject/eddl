/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
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

__global__ void gpu_low_mem_conv3D(int batch_size,
        int channels, int image_depth, int image_rows, int image_cols, const float *__restrict__ image,
        int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, const float *__restrict__ kernel,
        int out_depth, int out_rows, int out_cols, float *__restrict__ output,
        int pad_depth, int pad_row, int pad_col,
        int stride_depth, int stride_rows, int stride_cols)
{
    /* for (int b = 0; b < batch_size; b++)
    for (int nk = 0; nk < num_kernels; nk++)
    for (int k = 0; k < out_depth; k++)
    for (int i = 0; i < out_rows; i++)
    for (int j = 0; j < out_cols; j++)
    for (int c = 0; c < channels; c++) */
    int tid = blockIdx.x * low_mem_block_size + threadIdx.x;
    int threads = num_kernels * out_depth * out_rows * out_cols;
    if (tid >= threads) return;
    int j = tid % out_cols; tid /= out_cols;
    int i = tid % out_rows; tid /= out_rows;
    int k = tid % out_depth; tid /= out_depth;
    int nk = tid % num_kernels;
    int b = blockIdx.y;

    float s = 0;
    for (int z = 0; z < kernel_depth; z++) {
        int pz = k * stride_depth + z - pad_depth;
        if (pz >= 0 && pz < image_depth)
        for (int x = 0; x < kernel_rows; x++) {
            int px = i * stride_rows + x - pad_row;
            if (px >= 0 && px < image_rows)
            for (int y = 0; y < kernel_cols; y++) {
                int py = j * stride_cols + y - pad_col;
                if (py >= 0 && py < image_cols) {
                    for (int c = 0; c < channels; c++)
                        s += kernel[(((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols + y]
                           * image[(((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py];
                }
            }
        }
    }
    output[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j] = s;
}

__global__ void gpu_low_mem_conv3D_grad(int batch_size,
        int channels, int image_depth, int image_rows, int image_cols, const float *__restrict__ image,
        int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, float *__restrict__ kernel,
        int out_depth, int out_rows, int out_cols, const float *__restrict__ delta,
        int pad_depth, int pad_row, int pad_col,
        int stride_depth, int stride_rows, int stride_cols)
{
    /* for (int b = 0; b < batch_size; b++)
    for (int nk = 0; nk < num_kernels; nk++)
    for (int c = 0; c < channels; c++)
    for (int z = 0; z < kernel_depth; z++)
    for (int x = 0; x < kernel_rows; x++)
    for (int y = 0; y < kernel_cols; y++) */
    int tid = blockIdx.x * low_mem_block_size + threadIdx.x;
    int threads = num_kernels * channels * kernel_depth * kernel_rows * kernel_cols;
    if (tid >= threads) return;
    int y = tid % kernel_cols; tid /= kernel_cols;
    int x = tid % kernel_rows; tid /= kernel_rows;
    int z = tid % kernel_depth; tid /= kernel_depth;
    int c = tid % channels; tid /= channels;
    int nk = tid % num_kernels;
    int b = blockIdx.y;

    float s = 0.0;
    for (int k = 0; k < out_depth; k++) {
        int pz = k * stride_depth - pad_depth + z;
        if (pz >= 0 && pz < image_depth)
        for (int i = 0; i < out_rows; i++) {
            int px = i * stride_rows - pad_row + x;
            if (px >= 0 && px < image_rows)
            for (int j = 0; j < out_cols; j++) {
                int py = j * stride_cols - pad_col + y;
                if (py >= 0 && py < image_cols)
                s += image[(((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py] *
                    delta[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j];
            }
        }
    }
    atomicAdd(kernel + ((((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols) + y, s);
}

__global__ void gpu_low_mem_conv3D_back(int batch_size,
        int channels, int image_depth, int image_rows, int image_cols, float *__restrict__ image,
        int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, const float *__restrict__ kernel,
        int out_depth, int out_rows, int out_cols, const float *__restrict__ delta,
        int pad_depth, int pad_row, int pad_col,
        int stride_depth, int stride_rows, int stride_cols)
{
    /* for (int b = 0; b < batch_size; b++)
    for (int c = 0; c < channels; c++)
    for (int k = 0; k < out_depth; k++)
    for (int i = 0; i < out_rows; i++)
    for (int j = 0; j < out_cols; j++) */
    int tid = blockIdx.x * low_mem_block_size + threadIdx.x;
    int threads = channels * out_depth * out_rows * out_cols;
    if (tid >= threads) return;
    int j = tid % out_cols; tid /= out_cols;
    int i = tid % out_rows; tid /= out_rows;
    int k = tid % out_depth; tid /= out_depth;
    int c = tid;
    int b = blockIdx.y;

    for (int z = 0; z < kernel_depth; z++) {
        int pz = k * stride_depth + z - pad_depth;
        if (pz >= 0 && pz < image_depth)
        for (int x = 0; x < kernel_rows; x++) {
            int px = i * stride_rows - pad_row + x;
            if (px >= 0 && px < image_rows)
            for (int y = 0; y < kernel_cols; y++) {
                int py = j * stride_cols - pad_col + y;
                if (py >= 0 && py < image_cols) {
                    float s = 0.0;
                    for (int nk = 0; nk < num_kernels; nk++)
                        s += delta[(((b * num_kernels + nk) * out_depth + k) * out_rows + i) * out_cols + j]
                           * kernel[(((nk * channels + c) * kernel_depth + z) * kernel_rows + x) * kernel_cols + y];
                    atomicAdd(image + (((b * channels + c) * image_depth + pz) * image_rows + px) * image_cols + py, s);
                }
            }
        }
    }
}
