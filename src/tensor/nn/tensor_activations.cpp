/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/profiling.h"

#ifdef cFPGA
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#endif

#ifdef cGPU
#include "eddl/hardware/gpu/gpu_tensor.h"
#include "eddl/hardware/gpu/gpu_hw.h"
#include "eddl/hardware/gpu/nn/gpu_tensor_nn.h"
#endif

PROFILING_ENABLE_EXTERN(ReLu);
PROFILING_ENABLE_EXTERN(D_ReLu);
PROFILING_ENABLE_EXTERN(ThresholdedReLu);
PROFILING_ENABLE_EXTERN(LeakyReLu);
PROFILING_ENABLE_EXTERN(D_ThresholdedReLu);
PROFILING_ENABLE_EXTERN(D_LeakyReLu);
PROFILING_ENABLE_EXTERN(ELu);
PROFILING_ENABLE_EXTERN(D_ELu);
PROFILING_ENABLE_EXTERN(Sigmoid);
PROFILING_ENABLE_EXTERN(D_Sigmoid);
PROFILING_ENABLE_EXTERN(HardSigmoid);
PROFILING_ENABLE_EXTERN(D_HardSigmoid);
PROFILING_ENABLE_EXTERN(Tanh);
PROFILING_ENABLE_EXTERN(D_Tanh);
PROFILING_ENABLE_EXTERN(Softmax);
PROFILING_ENABLE_EXTERN(D_Softmax);
PROFILING_ENABLE_EXTERN(Exp);
PROFILING_ENABLE_EXTERN(D_Exp);
PROFILING_ENABLE_EXTERN(Linear);
PROFILING_ENABLE_EXTERN(D_Linear);
PROFILING_ENABLE_EXTERN(Softsign);
PROFILING_ENABLE_EXTERN(D_softsign);
PROFILING_ENABLE_EXTERN(Softplus);
PROFILING_ENABLE_EXTERN(D_softplus);

namespace tensorNN {

// ReLU
    void ReLu(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::ReLu");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::ReLu");

	    PROFILING_HEADER(ReLu);

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
            fpga_relu(A,B);
        }
#endif

        B->tsem->unlock();

	PROFILING_FOOTER(ReLu);
    }

// RELU Derivative, always increment over parent delta
    void D_ReLu(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device)) {
            msg("Tensors in different devices", "Tensor::D_ReLu");
        }
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

        PROFILING_HEADER(D_ReLu);

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
        fpga_d_relu(D,I,PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_ReLu);
    }

// ThresholdedReLu
    void ThresholdedReLu(Tensor *A, Tensor *B, float param) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::ThresholdedReLu");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::ThresholdedReLu");

        PROFILING_HEADER(ThresholdedReLu);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_thresholded_relu(A, B, param);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_thresholded_relu(A,B,param);
          }
#endif
#ifdef cFPGA
    else {
        fpga_thresholded_relu(A,B,param);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(ThresholdedReLu);
    }

// ThresholdedReLu Derivative
    void D_ThresholdedReLu(Tensor *D, Tensor *I, Tensor *PD, float param) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_ThresholdedReLu");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_ThresholdedReLu");

        PROFILING_HEADER(D_ThresholdedReLu);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_thresholded_relu(D, I, PD, param);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_thresholded_relu(D,I,PD,param);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_thresholded_relu(D, I, PD, param);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_ThresholdedReLu);
    }

// LeakyReLU
    void LeakyReLu(Tensor *A, Tensor *B, float param) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::LeakyReLu");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::LeakyReLu");

        PROFILING_HEADER(LeakyReLu);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_leaky_relu(A, B, param);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_leaky_relu(A,B,param);
          }
#endif
#ifdef cFPGA
    else {
        fpga_leaky_relu(A,B,param);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(LeakyReLu);
    }

// RELU Derivative, always increment over parent delta
    void D_LeakyReLu(Tensor *D, Tensor *I, Tensor *PD, float param) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_ReLu");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_ReLu");

        PROFILING_HEADER(D_LeakyReLu);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_leaky_relu(D, I, PD, param);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_leaky_relu(D,I,PD,param);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_leaky_relu(D,I,PD,param);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_LeakyReLu);
    }


// ELU
    void ELu(Tensor *A, Tensor *B, float param) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::ELu");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::ELu");

        PROFILING_HEADER(ELu);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_elu(A, B, param);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_elu(A,B ,param);
          }
#endif
#ifdef cFPGA
    else {
        fpga_elu(A,B,param);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(ELu);
    }

// ELU Derivative
    void D_ELu(Tensor *D, Tensor *I, Tensor *PD, float param) {
        if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_ELu");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_ELu");

        PROFILING_HEADER(D_ELu);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_elu(D, I, PD, param);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_elu(D, I, PD, param);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_elu(D, I, PD, param);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_ELu);
    }


// Softplus
    void Softplus(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softplus");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Softplus");

        PROFILING_HEADER(Softplus);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_softplus(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_softplus(A, B);
          }
#endif
#ifdef cFPGA
    else {
        fpga_softplus(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Softplus);
    }

// Softplus Derivative
    void D_softplus(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_softplus");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_softplus");

        PROFILING_HEADER(D_softplus);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_softplus(D, I, PD);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_softplus(D, I, PD);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_softplus(D, I, PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_softplus);
    }


// Softsign
    void Softsign(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softsign");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Softsign");

        PROFILING_HEADER(Softsign);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_softsign(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_softsign(A,B);
          }
#endif
#ifdef cFPGA
    else {
        fpga_softsign(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Softsign);
    }

// Softsign Derivative
    void D_softsign(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_softsign");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_softsign");

        PROFILING_HEADER(D_softsign);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_softsign(D, I, PD);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_softsign(D, I, PD);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_softsign(D, I, PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_softsign);
    }

// Linear
    void Linear(Tensor *A, Tensor *B, float param) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Linear");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Linear");

        PROFILING_HEADER(Linear);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_linear(A, B, param);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_linear(A,B ,param);
          }
#endif
#ifdef cFPGA
    else {
        fpga_linear(A, B, param);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Linear);
    }

// Linear Derivative
    void D_Linear(Tensor *D, Tensor *I, Tensor *PD, float param) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_Linear");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_Linear");

        PROFILING_HEADER(D_Linear);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_linear(D, I, PD, param);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_linear(D, I, PD, param);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_linear(D, I, PD, param);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_Linear);

    }

// Sigmoid
    void Sigmoid(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Sigmoid");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Sigmoid");

        PROFILING_HEADER(Sigmoid);

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
        fpga_sigmoid(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Sigmoid);
    }

// Sigmoid Derivative, always increment over parent delta
    void D_Sigmoid(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_Sigmoid");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_Sigmoid");

        PROFILING_HEADER(D_Sigmoid);

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
        fpga_d_sigmoid(D, I, PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_Sigmoid);
    }

// Hard Sigmoid
    void HardSigmoid(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::HardSigmoid");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::HardSigmoid");

        PROFILING_HEADER(HardSigmoid);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_hard_sigmoid(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_hard_sigmoid(A,B);
          }
#endif
#ifdef cFPGA
    else {
        fpga_hard_sigmoid(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(HardSigmoid);
    }

// Hard Sigmoid Derivative
    void D_HardSigmoid(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_HardSigmoid");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_HardSigmoid");

        PROFILING_HEADER(D_HardSigmoid);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_hard_sigmoid(D, I, PD);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
            gpu_d_hard_sigmoid(D,I,PD);

          }
#endif
#ifdef cFPGA
    else {
        fpga_d_hard_sigmoid(D, I, PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_HardSigmoid);
    }

// Exponential
    void Exp(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Exp");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Exp");

        PROFILING_HEADER(Exp);

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_exp(A, B);
        }
#ifdef cGPU
        else if (A->isGPU())
          {
          gpu_exp(A,B);

          }
#endif
#ifdef cFPGA
    else {
        fpga_exp(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Exp);
    }

// Exponential Derivative
    void D_Exp(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device)) msg("Tensors in different devices", "Tensor::D_Exp");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_Exp");

        PROFILING_HEADER(D_Exp);

        PD->tsem->lock();
        if (D->isCPU()) {
            cpu_d_exp(D, I, PD);
        }
#ifdef cGPU
        else if (D->isGPU())
          {
             gpu_d_exp(D,I,PD);
          }
#endif
#ifdef cFPGA
    else {
        fpga_d_exp(D, I, PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_Exp);
    }

// Tanh
    void Tanh(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Tanh");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Tanh");

        PROFILING_HEADER(Tanh);

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
        fpga_tanh(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Tanh);
    }

// Tanh Derivative
    void D_Tanh(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_Tanh");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_Tanh");

        PROFILING_HEADER(D_Tanh);

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
        fpga_d_tanh(D, I, PD);
    }
#endif
        PD->tsem->unlock();

        PROFILING_FOOTER(D_Tanh);
    }


// SOFTMAX
    void Softmax(Tensor *A, Tensor *B) {
        if (A->device != B->device) msg("Tensors in different devices", "Tensor::Softmax");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::Softmax");
        if (A->ndim != 2) msg("Softmax only over 2D Tensor (batch x logits)", "Tensor::Softmax");

        PROFILING_HEADER(Softmax);

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
        fpga_softmax(A, B);
    }
#endif

        B->tsem->unlock();

        PROFILING_FOOTER(Softmax);
    }

// SOFTMAX DERIVATIVE
    void D_Softmax(Tensor *D, Tensor *I, Tensor *PD) {
        if ((D->device != I->device) || (D->device != PD->device))
            msg("Tensors in different devices", "Tensor::D_Softmax");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_Softmax");
        if (D->ndim != 2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)", "Tensor::D_Softmax");

        PROFILING_HEADER(D_Softmax);

        if (D->isCPU()) {
            cpu_d_softmax(D, I, PD);
        }
#ifdef cGPU
        else if (D->isGPU())
          {

            // TODO: This could be improved (missing "gpu_d_softmax")
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
        fpga_d_softmax(D, I, PD);
    }
#endif

    PROFILING_FOOTER(D_Softmax);

    }




    // FULL SOFTMAX
    void FullSoftmax(Tensor *A, Tensor *B) {
        if (!Tensor::sameDevice(A, B)) msg("Tensors in different devices", "Tensor::FullSoftmax");
        if (!Tensor::sameShape(A, B)) msg("Incompatible dims", "Tensor::FullSoftmax");

        B->tsem->lock();
        if (A->isCPU()) {
            cpu_full_softmax(A, B, 1, true);
        }
#ifdef cGPU
        else if (A->isGPU())
        {
            gpu_full_softmax(A, B, 1, true);
        }
#endif
#ifdef cFPGA
        else {
                        msg("Not Implemented Error", "FullSoftmax");

    }
#endif

        B->tsem->unlock();
    }

    // FULL SOFTMAX DERIVATIVE
    void D_FullSoftmax(Tensor *D, Tensor *I, Tensor *PD) {
        if (!Tensor::sameDevice(D, I) || !Tensor::sameDevice(D, PD))
            msg("Tensors in different devices", "Tensor::D_FullSoftmax");
        if ((!Tensor::sameShape(D, I)) || (!Tensor::sameShape(D, PD))) msg("Incompatible dims", "Tensor::D_FullSoftmax");
        if (D->ndim != 2) msg("D_Softmax only over 2D Tensor (batch x delta_probs)", "Tensor::D_Softmax");

        if (D->isCPU()) {
            cpu_d_full_softmax(D, I, PD, 1);
        }
#ifdef cGPU
        else if (D->isGPU())
        {
            gpu_d_full_softmax(D, I, PD, 1);
        }
#endif
#ifdef cFPGA
        else {
                        msg("Not Implemented Error", "D_FullSoftmax");
    }
#endif
    }


}

