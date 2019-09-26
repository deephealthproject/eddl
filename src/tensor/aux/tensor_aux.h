//
// Created by Salva Carrión on 26/09/2019.
//

#ifndef EDDL_TENSOR_AUX_H
#define EDDL_TENSOR_AUX_H

#include <stdio.h>
#include <vector>
#include <string>
#include <mutex>

#include <Eigen/Dense>

#include "../tensor.h"
#include "../../descriptors/descriptors.h"

// TODO: Take this out of here -------------------
// ***** Losses *****************************
void cent(Tensor *A, Tensor *B, Tensor *C);

// ***** Metrics *****************************
int accuracy(Tensor *A, Tensor *B);


// ***** Activations *****************************
void ReLu(Tensor *A, Tensor *B);
void D_ReLu(Tensor *D, Tensor *I, Tensor *PD);

void Softmax(Tensor *A, Tensor *B);
void D_Softmax(Tensor *D, Tensor *I, Tensor *PD);


// ***** Deep Learning *****************************
void Conv2D(ConvolDescriptor *D);
void Conv2D_grad(ConvolDescriptor *D);
void Conv2D_back(ConvolDescriptor *D);

void MPool2D(PoolDescriptor *D);
void MPool2D_back(PoolDescriptor *D);

#endif //EDDL_TENSOR_AUX_H
