#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>

#include "tensor.h"
#include "../utils.h"

#ifdef cGPU
#include "../hardware/gpu/tensor_cuda.h"
#include "../hardware/gpu/tensor_cuda_op.h"
#endif

using namespace std;


// ReLU
void Tensor::ReLu(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::ReLu");
    if (!eqsize(A, B)) msg("Incompatible dims", "Tensor::ReLu");

    B->tsem->lock();
    if (A->isCPU()) {

        for (int i = 0; i < A->size; i++) {
            if (A->ptr[i] > 0.0) B->ptr[i] = A->ptr[i];
            else B->ptr[i] = 0.0;
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
      gpu_relu(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}

// RELU Derivative, always increment over parent delta
void Tensor::D_ReLu(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_ReLu");
    if ((!eqsize(D, I)) || (!eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

    PD->tsem->lock();
    if (D->isCPU()) {

        for (int i = 0; i < D->size; i++) {
            if (I->ptr[i] > 0.0) PD->ptr[i] = D->ptr[i];
            else PD->ptr[i] = 0.0;
        }
    }
#ifdef cGPU
    else if (D->isGPU())
      {
        gpu_d_relu(D,I,PD);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    PD->tsem->unlock();
}


// SOFTMAX
void Tensor::Softmax(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softmax");
    if (!eqsize(A, B)) msg("Incompatible dims", "Tensor::Softmax");
    if (A->ndim != 2) msg("Softmax only over 2D Tensor (batch x logits)", "Tensor::Softmax");

    B->tsem->lock();

    if (A->isCPU()) {
        float max, sum;


        for (int i = 0; i < A->shape[0]; i++) {

            max = (*A->ptr2).col(i).maxCoeff();
            for (int j = 0; j < A->shape[1]; j++)
                (*B->ptr2)(j, i) = std::exp((*A->ptr2)(j, i) - max);

            sum = (*B->ptr2).col(i).sum();
            for (int j = 0; j < B->shape[1]; j++)
                (*B->ptr2)(j, i) = (*B->ptr2)(j, i) / sum;
        }
    }
#ifdef cGPU
    else if (A->isGPU())
      {
        gpu_softmax(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}


// SOFTMAX DERIVATIVE
void Tensor::D_Softmax(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Softmax");
    if ((!eqsize(D, I)) || (!eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_Softmax");
    if (D->ndim != 2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)", "Tensor::D_Softmax");


    if (D->isCPU()) {
        PD->tsem->lock();

        for (int i = 0; i < D->size; i++)
            PD->ptr[i] += D->ptr[i] * (I->ptr[i] * (1.0 - I->ptr[i]));

        PD->tsem->unlock();
    }
#ifdef cGPU
    else if (D->isGPU())
      {

        Tensor *aux=new Tensor(D->getShape(),D->device);
        aux->set(1.0);
        Tensor::add(1.0,aux,-1.0,I,aux,0);
        Tensor::el_mult(I,aux,aux,0);
        Tensor::el_mult(D,aux,PD,1);

        delete aux;
      }
#endif
#ifdef cFPGA
    else {

    }
#endif


}