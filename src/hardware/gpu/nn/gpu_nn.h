/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_GPU_NN_H
#define EDDL_GPU_NN_H

#include <cuda.h>
#include <curand.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "../../../tensor/tensor.h"
#include "../../../descriptors/descriptors.h"



// Aux

// Activations
void gpu_relu(Tensor *A,Tensor *B);
void gpu_d_relu(Tensor *D,Tensor *I,Tensor *PD);

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

void gpu_tanh(Tensor *A,Tensor *B);
void gpu_d_tanh(Tensor *D,Tensor *I,Tensor *PD);

void gpu_linear(Tensor *A,Tensor *B,float param);
void gpu_d_linear(Tensor *D,Tensor *I,Tensor *PD,float param);

void gpu_softmax(Tensor *A,Tensor *B);

// Losses
void gpu_cent(Tensor *A,Tensor *B,Tensor *C);

// Metrics
void gpu_accuracy(Tensor *A,Tensor *B,int *acc);

// Conv
void gpu_conv2D(ConvolDescriptor *D);
void gpu_conv2D_grad(ConvolDescriptor *D);
void gpu_conv2D_back(ConvolDescriptor *D);

// Pool
void gpu_mpool2D(PoolDescriptor *D);
void gpu_mpool2D_back(PoolDescriptor *D);

// Tensor
void gpu_repeat_nn(Tensor *A, Tensor *B, vector<int> size);
void gpu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size);

#endif //EDDL_GPU_NN_H
