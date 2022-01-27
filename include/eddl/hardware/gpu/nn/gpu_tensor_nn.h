/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_GPU_TENSOR_NN_H
#define EDDL_GPU_TENSOR_NN_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"



// Aux

// Activations
void gpu_relu(Tensor *A,Tensor *B);
void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD);

void gpu_thresholded_relu(Tensor *A,Tensor *B,float param);
void gpu_d_thresholded_relu(Tensor *D,Tensor *I,Tensor *PD,float param);

void gpu_leaky_relu(Tensor *A,Tensor *B,float param);
void gpu_d_leaky_relu(Tensor *D,Tensor *I,Tensor *PD,float param);

void gpu_elu(Tensor *A,Tensor *B,float param);
void gpu_d_elu(Tensor *D,Tensor *I,Tensor *PD,float param);

void gpu_softplus(Tensor *A,Tensor *B);
void gpu_d_softplus(Tensor *D,Tensor *I,Tensor *PD);

void gpu_softsign(Tensor *A,Tensor *B);
void gpu_d_softsign(Tensor *D,Tensor *I,Tensor *PD);

void gpu_sigmoid(Tensor *A,Tensor *B);
void gpu_d_sigmoid(Tensor *D,Tensor *I,Tensor *PD);

void gpu_hard_sigmoid(Tensor *A,Tensor *B);
void gpu_d_hard_sigmoid(Tensor *D,Tensor *I,Tensor *PD);

void gpu_exp(Tensor *A,Tensor *B);
void gpu_d_exp(Tensor *D,Tensor *I,Tensor *PD);

void gpu_tanh(Tensor *A,Tensor *B);
void gpu_d_tanh(Tensor *D,Tensor *I,Tensor *PD);

void gpu_softmax(Tensor *A,Tensor *B);
//void gpu_d_softmax(Tensor *D,Tensor *I,Tensor *PD);  // Missing. Written as a chain of tensor operations

void gpu_full_softmax(Tensor *A, Tensor *B, int axis, bool stable);
void gpu_full_softmax_batched(Tensor *A, Tensor *B, bool stable);  // Aux. temp.
void gpu_full_softmax_nd(Tensor *A, Tensor *B, int axis, bool stable);  // Aux. temp.

void gpu_d_full_softmax(Tensor *D, Tensor *I, Tensor *PD, int axis);
void gpu_d_full_softmax_batched(Tensor *D, Tensor *I, Tensor *PD);  // Aux. temp.
void gpu_d_full_softmax_nd(Tensor *D, Tensor *I, Tensor *PD, int axis);  // Aux. temp.

void gpu_linear(Tensor *A,Tensor *B,float param);
void gpu_d_linear(Tensor *D,Tensor *I,Tensor *PD,float param);

// Losses
void gpu_cent(Tensor *A,Tensor *B,Tensor *C);

float gpu_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred);
void gpu_d_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta);

float gpu_binary_cross_entropy(Tensor* y_true, Tensor* y_pred);
void gpu_d_binary_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta);

// Metrics
void gpu_accuracy(Tensor *A,Tensor *B,int *acc);
void gpu_bin_accuracy(Tensor *A,Tensor *B,int *acc);

// Conv2D
void gpu_conv2D(ConvolDescriptor *D);
void gpu_conv2D_grad(ConvolDescriptor *D);
void gpu_conv2D_back(ConvolDescriptor *D);

// Conv3D
void gpu_conv3D(ConvolDescriptor3D *D);
void gpu_conv3D_grad(ConvolDescriptor3D *D);
void gpu_conv3D_back(ConvolDescriptor3D *D);

// ConvT3D
void gpu_convT2D(ConvolDescriptorT2D *D);
void gpu_convT2D_grad(ConvolDescriptorT2D *D);
void gpu_convT2D_back(ConvolDescriptorT2D *D);

// ConvT3D
void gpu_convT3D(ConvolDescriptorT3D *D);
void gpu_convT3D_grad(ConvolDescriptorT3D *D);
void gpu_convT3D_back(ConvolDescriptorT3D *D);

// MaxPool2D
void gpu_mpool2D(PoolDescriptor *D);
void gpu_mpool2D_back(PoolDescriptor *D);

// MaxPool3D
void gpu_mpool3D(PoolDescriptor3D *D);
void gpu_mpool3D_back(PoolDescriptor3D *D);

// AvgPool
void gpu_avgpool2D(PoolDescriptor *D);
void gpu_avgpool2D_back(PoolDescriptor *D);
// AvgPool3D
void gpu_avgpool3D(PoolDescriptor3D *D);
void gpu_avgpool3D_back(PoolDescriptor3D *D);

// Tensor
void gpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size);
void gpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size);

void gpu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);

void gpu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);

void gpu_expand_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd);
void gpu_expand_back_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd);

// BN
void gpu_permute_channels_first(Tensor *A,Tensor *B);
void gpu_permute_channels_last(Tensor *A,Tensor *B);
void gpu_permute_batch_first(Tensor *A,Tensor *B);
void gpu_permute_batch_last(Tensor *A,Tensor *B);

//quantize
void gpu_quantize_linear(Tensor *A, Tensor *B, float y_scale, int y_zero_point);
void gpu_dequantize_linear(Tensor *A, Tensor *B, float x_scale, int x_zero_point);

// new batchnorm implementation
void gpu_batchnorm_forward(int gpu_device, int b, int z, int rc,
        float *input, float *output, float *opa,
        float *global_mean, float *global_variance,
        float *affine_g, float *affine_b,
        float *mean, float *variance,
        bool trmode, float epsilon, float momentum);

void gpu_batchnorm_backward(int gpu_device, int b, int z, int rc,
        float *delta, float *opa, float *pdelta, float *gbn_g, float *gbn_b,
        float *bn_g, float *variance, float *mean1, float *mean2);

#endif //EDDL_GPU_TENSOR_NN_H
