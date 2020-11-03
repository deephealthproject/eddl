/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
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
void gpu_d_full_softmax(Tensor *D, Tensor *I, Tensor *PD, int axis);
void gpu_d_full_softmax_batched(Tensor *D, Tensor *I, Tensor *PD);  // Aux. temp.

void gpu_linear(Tensor *A,Tensor *B,float param);
void gpu_d_linear(Tensor *D,Tensor *I,Tensor *PD,float param);

// Losses
void gpu_cent(Tensor *A,Tensor *B,Tensor *C);

// Metrics
void gpu_accuracy(Tensor *A,Tensor *B,int *acc);
void gpu_bin_accuracy(Tensor *A,Tensor *B,int *acc);

// Conv
void gpu_conv2D(ConvolDescriptor *D);
void gpu_conv2D_grad(ConvolDescriptor *D);
void gpu_conv2D_back(ConvolDescriptor *D);

// MaxPool
void gpu_mpool2D(PoolDescriptor *D);
void gpu_mpool2D_back(PoolDescriptor *D);

// AvgPool
void gpu_avgpool2D(PoolDescriptor *D);
void gpu_avgpool2D_back(PoolDescriptor *D);

// Tensor
void gpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size);
void gpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size);

void gpu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void gpu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);

// BN
void gpu_permute_channels_first(Tensor *A,Tensor *B);
void gpu_permute_channels_last(Tensor *A,Tensor *B);
void gpu_permute_batch_first(Tensor *A,Tensor *B);
void gpu_permute_batch_last(Tensor *A,Tensor *B);

#endif //EDDL_GPU_TENSOR_NN_H
