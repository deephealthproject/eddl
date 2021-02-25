/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#ifndef EDDL_TENSOR_NN_H
#define EDDL_TENSOR_NN_H

#include "eddl/tensor/tensor.h"
#include "eddl/descriptors/descriptors.h"

namespace tensorNN{

// ***** Losses *****************************
    void cent(Tensor *A, Tensor *B, Tensor *C);

// ***** Metrics *****************************
    int accuracy(Tensor *A, Tensor *B);
    int bin_accuracy(Tensor *A, Tensor *B);


// ***** Activations *****************************
// ReLu
    void ReLu(Tensor *A, Tensor *B);
    void D_ReLu(Tensor *D, Tensor *I, Tensor *PD);

    void LeakyReLu(Tensor *A, Tensor *B,float param);
    void D_LeakyReLu(Tensor *D, Tensor *I, Tensor *PD,float param);

//ELU
    void ELu(Tensor *A, Tensor *B, float param);
    void D_ELu(Tensor *D, Tensor *I, Tensor *PD, float param);

//Thresholded ReLu
    void ThresholdedReLu(Tensor *A, Tensor *B, float param);
    void D_ThresholdedReLu(Tensor *D, Tensor *I, Tensor *PD, float param);

// Softplus
    void Softplus(Tensor *A, Tensor *B);
    void D_softplus(Tensor *D, Tensor *I, Tensor *PD);

// Softsign
    void Softsign(Tensor *A, Tensor *B);
    void D_softsign(Tensor *D, Tensor *I, Tensor *PD);

// Sigmoid
    void Sigmoid(Tensor *A, Tensor *B);
    void D_Sigmoid(Tensor *D, Tensor *I, Tensor *PD);

// Hard Sigmoid
    void HardSigmoid(Tensor *A, Tensor *B);
    void D_HardSigmoid(Tensor *D, Tensor *I, Tensor *PD);

// Exponential
    void Exp(Tensor *A, Tensor *B);
    void D_Exp(Tensor *D, Tensor *I, Tensor *PD);

// Softmax
    void Softmax(Tensor *A, Tensor *B);
    void D_Softmax(Tensor *D, Tensor *I, Tensor *PD);

// Full Softmax
    void FullSoftmax(Tensor *A, Tensor *B);
    void D_FullSoftmax(Tensor *D, Tensor *I, Tensor *PD);

// Tanh
    void Tanh(Tensor *A, Tensor *B);
    void D_Tanh(Tensor *D, Tensor *I, Tensor *PD);

//Linear
    void Linear(Tensor *A, Tensor *B, float param);
    void D_Linear(Tensor *D, Tensor *I, Tensor *PD, float param);

// ***** Deep Learning *****************************
// Conv2D
    void Conv2D(ConvolDescriptor *D);
    void Conv2DReLU(ConvolDescriptor *D);
    void Conv2D_grad(ConvolDescriptor *D);
    void Conv2D_back(ConvolDescriptor *D);

// MaxPool
    void MPool2D(PoolDescriptor *D);
    void MPool2D_back(PoolDescriptor *D);


// AvgPool
    void AvgPool2D(PoolDescriptor *D);
    void AvgPool2D_back(PoolDescriptor *D);

// ***** Tensor operations *****************************
    void repeat_nn(Tensor *A, Tensor *B, vector<int> size);
    void d_repeat_nn(Tensor *D, Tensor *P, vector<int> size);

    void select(Tensor *A, Tensor* B, SelDescriptor *sd);
    void select_back(Tensor *A, Tensor* B, SelDescriptor *sd);
    void set_select(Tensor *A, Tensor *B, SelDescriptor *sd);
    void set_select_back(Tensor *A, Tensor* B, SelDescriptor *sd);

// ***** Permutations for BatchNorm ********************
    void permute_channels_last(Tensor *A,Tensor *B);
    void permute_channels_first(Tensor *A,Tensor *B);
    void permute_batch_last(Tensor *A,Tensor *B);
    void permute_batch_first(Tensor *A,Tensor *B);

}

#endif //EDDL_TENSOR_NN_H
