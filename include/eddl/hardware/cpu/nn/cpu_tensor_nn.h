/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_CPU_TENSOR_NN_H
#define EDDL_CPU_TENSOR_NN_H

#include "eddl/hardware/cpu/cpu_profile.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

// Aux
float get_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize);
void add_pixel(int b,int px,int py,int pz,ConvolDescriptor *D,int isize,int irsize,float val);

// Activations
void cpu_relu(Tensor *A, Tensor *B);
void cpu_d_relu(Tensor *D, Tensor *I, Tensor *PD);

void cpu_thresholded_relu(Tensor *A, Tensor *B, float param);
void cpu_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD, float param);

void cpu_leaky_relu(Tensor *A, Tensor *B, float param);
void cpu_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD, float param);

void cpu_elu(Tensor *A, Tensor *B, float param);
void cpu_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param);

void cpu_softplus(Tensor *A, Tensor *B);
void cpu_d_softplus(Tensor *D, Tensor *I, Tensor *PD);

void cpu_softsign(Tensor *A, Tensor *B);
void cpu_d_softsign(Tensor *D, Tensor *I, Tensor *PD);

void cpu_sigmoid(Tensor *A, Tensor *B);
void cpu_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD);

void cpu_hard_sigmoid(Tensor *A, Tensor *B);
void cpu_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD);

void cpu_exp(Tensor *A, Tensor *B);
void cpu_d_exp(Tensor *D, Tensor *I, Tensor *PD);

void cpu_tanh(Tensor *A, Tensor *B);
void cpu_d_tanh(Tensor *D, Tensor *I, Tensor *PD);

void cpu_softmax(Tensor *A, Tensor *B);
void cpu_d_softmax(Tensor *D, Tensor *I, Tensor *PD);

void cpu_full_softmax(Tensor *A, Tensor *B, int axis, bool stable);
void cpu_full_softmax_batched_2d(Tensor *A, Tensor *B, bool stable);  // TODO: Temp. function
void cpu_full_softmax_nd(Tensor *A, Tensor *B, int axis, bool stable);  // TODO: Temp. function
void cpu_d_full_softmax(Tensor *D, Tensor *I, Tensor *PD, int axis);
void cpu_d_full_softmax_batched_2d(Tensor *D, Tensor *I, Tensor *PD);  // TODO: Temp. function
void cpu_d_full_softmax_nd(Tensor *D, Tensor *I, Tensor *PD, int axis); // TODO: Temp. function

void cpu_linear(Tensor *A, Tensor *B, float param);
void cpu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param);


// Losses
void cpu_cent(Tensor *A, Tensor *B, Tensor *C);
void cpu_bin_cent(Tensor *A, Tensor *B, Tensor *C);

float cpu_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred);
void cpu_d_categorical_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta);

float cpu_binary_cross_entropy(Tensor* y_true, Tensor* y_pred);
void cpu_d_binary_cross_entropy(Tensor* y_true, Tensor* y_pred, Tensor* delta);


// Metrics
int cpu_accuracy(Tensor *A, Tensor *B);
int cpu_bin_accuracy(Tensor *A, Tensor *B);


// Conv2D
void cpu_conv2D(ConvolDescriptor *D);
void cpu_conv2D_grad(ConvolDescriptor *D);
void cpu_conv2D_back(ConvolDescriptor *D);

// Conv3D
void cpu_conv3D(ConvolDescriptor3D *D);
void cpu_conv3D_grad(ConvolDescriptor3D *D);
void cpu_conv3D_back(ConvolDescriptor3D *D);

// MaxPool2D
void cpu_mpool2D(PoolDescriptor*D);
void cpu_mpool2D_back(PoolDescriptor *D);

// MaxPool3D
void cpu_mpool3D(PoolDescriptor3D *D);
void cpu_mpool3D_back(PoolDescriptor3D *D);

// AvgPool
void cpu_avgpool2D(PoolDescriptor*D);
void cpu_avgpool2D_back(PoolDescriptor *D);

// Tensor (special functions that deal with 4D tensors)
void cpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size);
void cpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size);

void cpu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void cpu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void cpu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);

void cpu_expand_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd);
void cpu_expand_back_nn(Tensor *A, Tensor *B, ExpandDescriptor *sd);

void cpu_repeat_batch(Tensor *A, Tensor *B);

// BN
void cpu_permute_channels_first(Tensor *A,Tensor *B);
void cpu_permute_channels_last(Tensor *A,Tensor *B);
void cpu_permute_batch_first(Tensor *A,Tensor *B);
void cpu_permute_batch_last(Tensor *A,Tensor *B);
// new batchnorm implementation
void cpu_batchnorm_forward(int b, int z, int rc,
        float *input, float *output, float *opa,
        float *global_mean, float *global_variance,
        float *affine_g, float *affine_b,
        float *mean, float *variance,
        bool trmode, float epsilon, float momentum);

void cpu_batchnorm_backward(int b, int z, int rc,
        float *delta, float *opa, float *pdelta, float *gbn_g,
        float *gbn_b, float *bn_g, float *variance,
        float *mean1, float *mean2);
#endif //EDDL_CPU_TENSOR_NN_H
