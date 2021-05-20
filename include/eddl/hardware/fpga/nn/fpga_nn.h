/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#ifndef EDDL_FPGA_NN_H
#define EDDL_FPGA_NN_H

#include "eddl/hardware/fpga/fpga_profile.h"

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

// Activations
void fpga_relu(Tensor *A, Tensor *B);
void fpga_d_relu(Tensor *D, Tensor *I, Tensor *PD);

void fpga_thresholded_relu(Tensor *A, Tensor *B, float param);
void fpga_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD, float param);

void fpga_leaky_relu(Tensor *A, Tensor *B, float param);
void fpga_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD, float param);

void fpga_elu(Tensor *A, Tensor *B, float param);
void fpga_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param);

void fpga_softplus(Tensor *A, Tensor *B);
void fpga_d_softplus(Tensor *D, Tensor *I, Tensor *PD);

void fpga_softsign(Tensor *A, Tensor *B);
void fpga_d_softsign(Tensor *D, Tensor *I, Tensor *PD);

void fpga_sigmoid(Tensor *A, Tensor *B);
void fpga_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD);

void fpga_hard_sigmoid(Tensor *A, Tensor *B);
void fpga_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD);

void fpga_exp(Tensor *A, Tensor *B);
void fpga_d_exp(Tensor *D, Tensor *I, Tensor *PD);

void fpga_tanh(Tensor *A, Tensor *B);
void fpga_d_tanh(Tensor *D, Tensor *I, Tensor *PD);

void fpga_full_softmax(Tensor *A, Tensor *B, int axis, bool stable);

void fpga_softmax(Tensor *A, Tensor *B);
void fpga_d_softmax(Tensor *D, Tensor *I, Tensor *PD);

void fpga_linear(Tensor *A, Tensor *B, float param);
void fpga_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param);

// Losses
void fpga_cent(Tensor *A, Tensor *B, Tensor *C);

// Metrics
int fpga_accuracy(Tensor *A, Tensor *B);
int fpga_bin_accuracy(Tensor *A, Tensor *B);

// Conv
void fpga_conv2D(ConvolDescriptor *D);
void fpga_conv2DReLU(ConvolDescriptor *D);
void fpga_conv2D_grad(ConvolDescriptor *D);
void fpga_conv2D_back(ConvolDescriptor *D);
void fpga_reshape_kernel_data_convol(ConvolDescriptor *D, int KW, int KH, int I, int O, int CPI, int CPO);
int  fpga_conv2D_v1X(ConvolDescriptor *D, int enable_relu);
int  fpga_conv2D_v2X(ConvolDescriptor *D, int enable_relu);

// MaxPool
void fpga_mpool2D(PoolDescriptor*D);
void fpga_mpool2D_back(PoolDescriptor *D);

// AvgPool
void fpga_avgpool2D(PoolDescriptor*D);
void fpga_avgpool2D_back(PoolDescriptor *D);

// Tensor (special functions that deal with 4D tensors)
void fpga_repeat_nn(Tensor *A, Tensor *B, vector<int> size);
void fpga_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size);

void fpga_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd);
void fpga_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd);

// BN
void fpga_permute_channels_first(Tensor *A,Tensor *B);
void fpga_permute_channels_last(Tensor *A,Tensor *B);
void fpga_permute_batch_first(Tensor *A,Tensor *B);
void fpga_permute_batch_last(Tensor *A,Tensor *B);

//Fused
void fpga_conv_stm(ConvolDescriptor *D);
void fpga_conv_stm_transform(ConvolDescriptor *D);

void fpga_conv_maxpool(ConvolDescriptor *D);
void fpga_conv_maxpool_transform(ConvolDescriptor *D);
#endif //EDDL_FPGA_NN_H
