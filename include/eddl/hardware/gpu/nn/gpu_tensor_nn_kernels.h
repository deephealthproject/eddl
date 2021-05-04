/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_GPU_TENSOR_NN_KERNELS_H
#define EDDL_GPU_TENSOR_NN_KERNELS_H


#include <cuda.h>
#include <cstdio>


// GPU: Activations
__global__ void relu(float *a,float *b,long int size);
__global__ void d_relu(float *d,float *i,float *pd,long int size);

__global__ void thresholded_relu(float *a,float *b,float param, long int size);
__global__ void d_thresholded_relu(float *d,float *i,float *pd,float param, long int size);

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

__global__ void hard_sigmoid(float *a,float *b,long int size);
__global__ void d_hard_sigmoid(float *d,float *i,float *pd,long int size);

__global__ void exp(float *a,float *b,long int size);
__global__ void d_exp(float *d,float *i,float *pd,long int size);

__global__ void tanh(float *a,float *b,long int size);
__global__ void d_tanh(float *d,float *i,float *pd,long int size);

__global__ void softmax(float* E,float* N,float* auxE ,long int sample_ndim, long int n_vals);  // TODO: DEPRECATED
//__global__ void d_softmax(float *d,float *i,float *pd,long int size);  // TODO: DEPRECATED

__global__ void full_softmax_batched(float *A, float *B, bool stable, unsigned int n_batches, unsigned int n_features);
__global__ void full_d_softmax_batched(float *D,float *I,float *PD, unsigned int n_batches, unsigned int n_features);
__global__ void full_softmax_nd(float *A, float *B, bool stable, int n_samples, int inner_stride, int sample_stride, int k_stride);
__global__ void full_d_softmax_nd(float *D,float *I,float *PD, int n_samples, int inner_stride, int sample_stride, int k_stride);

__global__ void linear(float *a,float *b,float param, long int size);
__global__ void d_linear(float *d,float *i,float *pd,float param, long int size);


// GPU: Losses
__global__ void cent(float* a, float* b, float* c, long int size);

__global__ void gpu_categorical_cross_entropy(float* y_true, float* y_pred, float* sum_array, unsigned int n_batches, unsigned int n_features);
__global__ void gpu_d_categorical_cross_entropy(float* y_true, float* y_pred, float* delta, long int size);

__global__ void gpu_binary_cross_entropy(float* y_true, float* y_pred, float* sum_array, unsigned int size);
__global__ void gpu_d_binary_cross_entropy(float* y_true, float* y_pred, float* delta, long int size);


// GPU: Metrics
__global__ void accuracy(float* T, float* N,float* acc,long int cols, long int total_ops, int* MC_err);
__global__ void bin_accuracy(float* T, float* N, int size, int* acc);


// GPU: Conv
__global__ void gpu_traspose_batch_depth(float *Bptr, float *ptr, int b,int z,int r, int c);
__global__ void gpu_addbias_k(float *O, int b, int r,int c,int nk,float *bias, int offset);
__global__ void gpu_deltabias_k(float *D, int batch, int r,int c,int nk,float *bias, int offset);
__global__ void gpu_im2col_k(float* I, float *ptrI, int b,int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int padrt,int padrb,int padcl,int padcr,int col2im);
__global__ void gpu_im2col_k_low(float* I, int b, float *ptrI, int irows,int icols, int idepth, float* K, int nk, int kr,int kc, float* O,int orows,int ocols,int sr,int sc,int padrt,int padrb,int padcl,int padcr,int col2im);

const int low_mem_block_size = 256;
__global__ void gpu_low_mem_conv3D(int batch_size, int channels, int image_depth, int image_rows, int image_cols, const float *image, int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, const float *kernel, int out_depth, int out_rows, int out_cols, float *output, int pad_depth, int pad_row, int pad_col, int stride_depth, int stride_rows, int stride_cols);
__global__ void gpu_low_mem_conv3D_grad(int batch_size, int channels, int image_depth, int image_rows, int image_cols, const float *image, int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, float *kernel, int out_depth, int out_rows, int out_cols, const float *delta, int pad_depth, int pad_row, int pad_col, int stride_depth, int stride_rows, int stride_cols);
__global__ void gpu_low_mem_conv3D_back(int batch_size, int channels, int image_depth, int image_rows, int image_cols, float *image, int num_kernels, int kernel_depth, int kernel_rows, int kernel_cols, const float *kernel, int out_depth, int out_rows, int out_cols, const float *delta, int pad_depth, int pad_row, int pad_col, int stride_depth, int stride_rows, int stride_cols);

// GPU: Pool
// MaxPool 2D
__global__ void maxpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt,int padrb,int padcl,int padcr, float* indX, float* indY);
__global__ void maxpool2d_back(float* D, float* ID, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr,float* indX, float* indY);

// MaxPool 3D
__global__ void maxpool3d(float* I, int batch, int ichannels, int idepth,int irows,int icols,  int kd, int kr, int kc, float* O, int ochannels, int odepth, int orows, int ocols,  int sd, int sr, int sc, int paddf, int paddb, int padrt, int padrb, int padcl, int padcr, float* indX, float* indY, float* indZ);
__global__ void maxpool3d_back(float* D, float* ID, int batch, int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr,float* indX, float* indY);

// AvgPool 2D
__global__ void avgpool2d(float* I, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr);
__global__ void avgpool2d_back(float* D, float* ID, int batch,int irows,int icols, int idepth, int kr,int kc, float* O,int orows,int ocols, int odepth, int sr,int sc,int padrt, int padrb,int padcl, int padcr);

// AvgPool 3D
__global__ void avgpool3d(float* I, int batch, int ichannels, int idepth,int irows,int icols,  int kd, int kr, int kc, float* O, int ochannels, int odepth, int orows, int ocols,  int sd, int sr, int sc, int paddf, int paddb, int padrt, int padrb, int padcl, int padcr, float* indX, float* indY, float* indZ);


// GPU: Tensor
__global__ void repeat_nn_k(float *a, int batch, int depth, int a_rows, int a_cols, float *b, int b_rows, int b_cols, int *size);
__global__ void d_repeat_nn_k(float *d, int batch, int depth, int d_rows, int d_cols, float *a, int a_rows, int a_cols, int *size);

__global__ void gpu_select_nn(float *A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str);
__global__ void gpu_select_back_nn(float *A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str);

__global__ void gpu_set_select_nn(float *A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str);
__global__ void gpu_set_select_back_nn(float *A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str);

__global__ void gpu_expand_nn(float *A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str);
__global__ void gpu_expand_back_nn(float *A, float* B, long int size, int* indices, int A_batch_str, int B_batch_str);

// BN
__global__ void bn_permute_channels_first(float *src, float *dest,int b,int z,int r,int c,long int size);
__global__ void bn_permute_channels_last(float *src, float *dest,int b,int z,int r,int c,long int size);
__global__ void bn_permute_batch_first(float *src, float *dest,int b,int z,int r,int c,long int size);
__global__ void bn_permute_batch_last(float *src, float *dest,int b,int z,int r,int c,long int size);

// new batchnorm implementation
const int batch_norm_block_size = 256;
__global__ void gpu_batchnorm_forward_1(int b, int rc, int rcz, float *input, float *mean, float *variance);
__global__ void gpu_batchnorm_forward_2(int z, float inv_N, float *mean, float *variance, float momentum, float *global_mean, float *global_variance, float epsilon);
__global__ void gpu_batchnorm_forward_3(int b, int rc, int rcz, float *input, float *mean, float *variance, float *affine_g, float *affine_b, float *opa, float *output);

__global__ void gpu_batchnorm_backward_1(int b, int rc, int rcz, float *delta, float *opa, float *bn_g, float *mean1, float *mean2);
__global__ void gpu_batchnorm_backward_2(int z, float inv_N, float *mean1, float *mean2, float *gbn_g, float *gbn_b, float *bn_g);
__global__ void gpu_batchnorm_backward_3(int b, int rc, int rcz, float *delta, float *opa, float *pdelta, float *mean1, float *mean2, float *variance);

#endif
