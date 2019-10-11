/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _TENSOR_CUDA_KERNELS_
#define _TENSOR_CUDA_KERNELS_


#include <cuda.h>


// GPU: Activations
__global__ void relu(float *a,float *b,long int size);
__global__ void d_relu(float *d,float *i,float *pd,long int size);
__global__ void softmax(float* E,float* N,float* auxE ,long int sample_ndim, long int n_vals);

// GPU: Losses
__global__ void cent(float* a, float* b, float* c, long int size);

// GPU: Metrics
__global__ void accuracy(float* T, float* N,float* acc,long int cols, long int total_ops, int* MC_err);

// GPU: Conv
__global__ void gpu_addbias_k(float *O, int b, int r,int c,int nk,float *bias);
__global__ void gpu_deltabias_k(float *D, int batch, int r,int c,int nk,float *bias);
__global__ void gpu_im2col_k(float* I, float *ptrI, int b,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int pad,int col2im);

// GPU: Pool
__global__ void maxpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padr, int padc, float* indX, float* indY);
__global__ void maxpool2d_back(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, int sr,int sc,int padr, int padc, float* indX, float* indY, float* D, float* ID);

// GPU: Tensor
__global__ void repeat_nn_k(float *a, int batch, int depth, int a_rows, int a_cols, float *b, int b_rows, int b_cols, int *size);
__global__ void d_repeat_nn_k(float *d, int batch, int depth, int d_rows, int d_cols, float *a, int a_rows, int a_cols, int *size);


#endif