/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef _TENSOR_CUDA_KERNELS_
#define _TENSOR_CUDA_KERNELS_


#include <cuda.h>
#include <stdio.h>

// todo
#define GPU_MAX_FLOAT 1000000.0f
#define GPU_MIN_FLOAT -10000000.0f

// GPU: Activations
__global__ void relu(float *a,float *b,long int size);
__global__ void d_relu(float *d,float *i,float *pd,long int size);

__global__ void leaky_relu(float *a,float *b,float param, long int size);
__global__ void d_leaky_relu(float *d,float *i,float *pd,float param, long int size);

__global__ void elu(float *a,float *b,float param, long int size);
__global__ void d_elu(float *d,float *i,float *pd,float param, long int size);

__global__ void softplus(float *a,float *b,long int size);
__global__ void d_softplus(float *d,float *i,float *pd,long int size);

__global__ void softsign(float *a,float *b,long int size);
__global__ void d_softsign(float *d,float *i,float *pd,long int size);

__global__ void sigmoid(float *a,float *b,long int size);
__global__ void d_sigmoid(float *d,float *i,float *pd,long int size);

__global__ void tanh(float *a,float *b,long int size);
__global__ void d_tanh(float *d,float *i,float *pd,long int size);

__global__ void linear(float *a,float *b,float param, long int size);
__global__ void d_linear(float *d,float *i,float *pd,float param, long int size);

__global__ void softmax(float* E,float* N,float* auxE ,long int sample_ndim, long int n_vals);

// GPU: Losses
__global__ void cent(float* a, float* b, float* c, long int size);

// GPU: Metrics
__global__ void accuracy(float* T, float* N,float* acc,long int cols, long int total_ops, int* MC_err);

// GPU: Conv
__global__ void gpu_addbias_k(float *O, int b, int r,int c,int nk,float *bias);
__global__ void gpu_deltabias_k(float *D, int batch, int r,int c,int nk,float *bias);
__global__ void gpu_im2col_k(float* I, float *ptrI, int b,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int padrt,int padrb,int padcl,int padcr,int col2im);
__global__ void gpu_im2col_k_low(float* I, int b, float *ptrI, int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int padrt,int padrb,int padcl,int padcr,int col2im);

// GPU: Pool
__global__ void maxpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt,int padrb,int padcl,int padcr, float* indX, float* indY);

//
__global__ void maxpool2d_back(float* D, float* ID, int batch, int irows, int icols, int orows, int ocols, int depth, float* indX, float* indY);


// GPU: Tensor
__global__ void repeat_nn_k(float *a, int batch, int depth, int a_rows, int a_cols, float *b, int b_rows, int b_cols, int *size);
__global__ void d_repeat_nn_k(float *d, int batch, int depth, int d_rows, int d_cols, float *a, int a_rows, int a_cols, int *size);


#endif
