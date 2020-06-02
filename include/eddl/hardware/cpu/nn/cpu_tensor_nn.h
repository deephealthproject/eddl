/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#ifndef EDDL_CPU_TENSOR_NN_H
#define EDDL_CPU_TENSOR_NN_H

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

void cpu_linear(Tensor *A, Tensor *B, float param);
void cpu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param);

// Losses
void cpu_cent(Tensor *A, Tensor *B, Tensor *C);
void cpu_bin_cent(Tensor *A, Tensor *B, Tensor *C);

// Metrics
int cpu_accuracy(Tensor *A, Tensor *B);
int cpu_bin_accuracy(Tensor *A, Tensor *B);


// Conv
void cpu_conv2D(ConvolDescriptor *D);
void cpu_conv2D_grad(ConvolDescriptor *D);
void cpu_conv2D_back(ConvolDescriptor *D);

// MaxPool
void cpu_mpool2D(PoolDescriptor*D);
void cpu_mpool2D_back(PoolDescriptor *D);

// AvgPool
void cpu_avgpool2D(PoolDescriptor*D);
void cpu_avgpool2D_back(PoolDescriptor *D);

// Tensor (special functions that deal with 4D tensors)
void cpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size);
void cpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size);

// BN
void cpu_permute_channels_first(Tensor *A,Tensor *B);
void cpu_permute_channels_last(Tensor *A,Tensor *B);
void cpu_permute_batch_first(Tensor *A,Tensor *B);
void cpu_permute_batch_last(Tensor *A,Tensor *B);
#endif //EDDL_CPU_TENSOR_NN_H
