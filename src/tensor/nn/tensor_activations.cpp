/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "tensor_nn.h"
#include "../../hardware/cpu/nn/cpu_nn.h"

#ifdef cGPU
#include "../../hardware/gpu/gpu_tensor.h"
#include "../../hardware/gpu/gpu_hw.h"
#include "../../hardware/gpu/nn/gpu_nn.h"
#endif


// ReLU
void ReLu(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::ReLu");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::ReLu");

    B->tsem->lock();
    if (A->isCPU()) {
        cpu_relu(A, B);
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
void D_ReLu(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_ReLu");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

    PD->tsem->lock();
    if (D->isCPU()) {
        cpu_d_relu(D, I, PD);
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

// LeakyReLU
void LReLu(Tensor *A, Tensor *B,float param) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::ReLu");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::ReLu");

    B->tsem->lock();
    if (A->isCPU()) {
        cpu_lrelu(A, B,param);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
      gpu_lrelu(A,B,param);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}

// RELU Derivative, always increment over parent delta
void D_LReLu(Tensor *D, Tensor *I, Tensor *PD,float param) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_ReLu");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

    PD->tsem->lock();
    if (D->isCPU()) {
        cpu_d_lrelu(D, I, PD,param);
    }
#ifdef cGPU
    else if (D->isGPU())
      {
        gpu_d_lrelu(D,I,PD,param);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    PD->tsem->unlock();
}


// Sigmoid
void Sigmoid(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::Sigmoid");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::Sigmoid");

    B->tsem->lock();
    if (A->isCPU()) {
        cpu_sigmoid(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
      gpu_sigmoid(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}

// Sigmoid Derivative, always increment over parent delta
void D_Sigmoid(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Sigmoid");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_Sigmoid");

    PD->tsem->lock();
    if (D->isCPU()) {
        cpu_d_sigmoid(D, I, PD);
    }
#ifdef cGPU
    else if (D->isGPU())
      {
        gpu_d_sigmoid(D,I,PD);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    PD->tsem->unlock();
}

// Sigmoid
void Tanh(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::Tanh");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::Tanh");

    B->tsem->lock();
    if (A->isCPU()) {
        cpu_tanh(A, B);
    }
#ifdef cGPU
    else if (A->isGPU())
      {
      gpu_tanh(A,B);
      }
#endif
#ifdef cFPGA
    else {

    }
#endif

    B->tsem->unlock();
}

// Sigmoid Derivative, always increment over parent delta
void D_Tanh(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Sigmoid");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_Sigmoid");

    PD->tsem->lock();
    if (D->isCPU()) {
        cpu_d_tanh(D, I, PD);
    }
#ifdef cGPU
    else if (D->isGPU())
      {
        gpu_d_tanh(D,I,PD);

      }
#endif
#ifdef cFPGA
    else {

    }
#endif
    PD->tsem->unlock();
}


// SOFTMAX
void Softmax(Tensor *A, Tensor *B) {
    if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softmax");
    if (!Tensor::eqsize(A, B)) msg("Incompatible dims", "Tensor::Softmax");
    if (A->ndim != 2) msg("Softmax only over 2D Tensor (batch x logits)", "Tensor::Softmax");

    B->tsem->lock();

    if (A->isCPU()) {
        cpu_softmax(A, B);
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
void D_Softmax(Tensor *D, Tensor *I, Tensor *PD) {
    if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Softmax");
    if ((!Tensor::eqsize(D, I)) || (!Tensor::eqsize(D, PD))) msg("Incompatible dims", "Tensor::D_Softmax");
    if (D->ndim != 2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)", "Tensor::D_Softmax");

    if (D->isCPU()) {
       cpu_d_softmax(D, I, PD);
    }
#ifdef cGPU
    else if (D->isGPU())
      {

        Tensor *aux=new Tensor(D->getShape(),D->device);
        aux->fill_(1.0);
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
