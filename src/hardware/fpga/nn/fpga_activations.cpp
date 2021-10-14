/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/fpga/fpga_hw.h"   // for buffer copies
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"  // for cpu emulation purposes

// prueba
extern cl::Context      context;

// -----------------------------------------------------------------
// relu
//
void fpga_cpuemu_relu(Tensor *A, Tensor *B){
    _profile(_CPU_RELU, 0);
    fpga_copy_from_fpga(A, A->ptr, 0);
    fpga_data_type *ptrA = (fpga_data_type *) A->ptr;
    fpga_data_type *ptrB = (fpga_data_type *) B->ptr;
#pragma omp parallel for
    for (int i = 0; i < A->size; i++) {
        if (ptrA[i] > fpga_data_type(0)) ptrB[i]= ptrA[i];
        else ptrB[i] = fpga_data_type(0);
    }
    fpga_copy_to_fpga(B->ptr, B, 0);

    _profile(_CPU_RELU, 1);
}

void fpga_relu(Tensor *A, Tensor *B){
  _debug_fpga_funcs("ReLU");
  _profile_fpga(_FPGA_RELU, 0);
  _profile_fpga_tensor(A);
  #ifndef K_ENABLED_RELU
  fpga_cpuemu_relu(A, B);
  #else
  
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_relu.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_relu.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_relu.setArg(2, A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_relu, NULL, &event));

  //  event.wait();
  q.finish();
#endif
  _profile_fpga_tensor(B);
  _profile_fpga(_FPGA_RELU, 1);
}

// -----------------------------------------------------------------
// d_relu
//
void fpga_cpuemu_d_relu(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_relu(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_relu(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_RELU, 0);
#ifndef K_ENABLED_D_RELU
  fpga_cpuemu_d_relu(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_relu.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_relu.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_relu.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_relu.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_relu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_RELU, 1);
}

// -----------------------------------------------------------------
// thbresholded_relu
//
void fpga_cpuemu_thresholded_relu(Tensor *A, Tensor *B, float param){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_thresholded_relu(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_thresholded_relu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_THRESHOLDED_RELU, 0);
#ifndef K_ENABLED_THRESHOLDED_RELU
  fpga_cpuemu_thresholded_relu(A, B, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_thresholded_relu.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_thresholded_relu.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_thresholded_relu.setArg(2, (long int)A->size));
  OCL_CHECK(err, err = kernel_thresholded_relu.setArg(3, param));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_relu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_THRESHOLDED_RELU, 1);
}

// -----------------------------------------------------------------
// d_thresholded_relu
//
void fpga_cpuemu_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_thresholded_relu(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_THRESHOLDED_RELU, 0);
#ifndef K_ENABLED_D_THRESHOLDED_RELU
  fpga_cpuemu_d_thresholded_relu(D, I, PD, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(3, (long int)D->size));
  OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(4, param));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_thresholded_relu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_THRESHOLDED_RELU, 1);
}

// -----------------------------------------------------------------
// leaky_relu
//
void fpga_cpuemu_leaky_relu(Tensor *A, Tensor *B, float param){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_leaky_relu(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_leaky_relu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_LEAKY_RELU, 0);
#ifndef K_ENABLED_LEAKY_RELU
  fpga_cpuemu_leaky_relu(A, B, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_leaky_relu.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_leaky_relu.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_leaky_relu.setArg(2, (long int)A->size));
  OCL_CHECK(err, err = kernel_leaky_relu.setArg(4, param));

  OCL_CHECK(err, err = q.enqueueTask(kernel_leaky_relu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_LEAKY_RELU, 1);
}

// -----------------------------------------------------------------
// d_leaky_relu
//
void fpga_cpuemu_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_leaky_relu(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_LEAKY_RELU, 0);
#ifndef K_ENABLED_D_LEAKY_RELU
  fpga_cpuemu_d_leaky_relu(D, I, PD, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(3, (long int)D->size));
  OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(4, param));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_leaky_relu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_LEAKY_RELU, 1);
}

// -----------------------------------------------------------------
// elu
//
void fpga_cpuemu_elu(Tensor *A, Tensor *B, float param){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_elu(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_elu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_ELU, 0);
#ifndef K_ENABLED_ELU
  fpga_cpuemu_elu(A, B, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_elu.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_elu.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_elu.setArg(3, (long int)A->size));
  OCL_CHECK(err, err = kernel_elu.setArg(4, param));

  OCL_CHECK(err, err = q.enqueueTask(kernel_elu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_ELU, 1);
}

// -----------------------------------------------------------------
// d_elu
//
void fpga_cpuemu_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_elu(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_ELU, 0);
#ifndef K_ENABLED_D_ELU
  fpga_cpuemu_d_elu(D, I, PD, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_elu.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_elu.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_elu.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_elu.setArg(3, (long int)D->size));
  OCL_CHECK(err, err = kernel_d_elu.setArg(4, param));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_elu, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_ELU, 1);
}

// -----------------------------------------------------------------
// softplus
//
void fpga_cpuemu_softplus(Tensor *A, Tensor *B){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_softplus(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_softplus(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SOFTPLUS, 0);
#ifndef K_ENABLED_SOFTPLUS
  fpga_cpuemu_softplus(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_softplus.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_softplus.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_softplus.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_softplus, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SOFTPLUS, 1);
}

// -----------------------------------------------------------------
// d_softplus
//
void fpga_cpuemu_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_softplus(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_SOFTPLUS, 0);
#ifndef K_ENABLED_D_SOFTPLUS
  fpga_cpuemu_d_softplus(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_softplus.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softplus.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softplus.setArg(2, (long int)D->size));
  OCL_CHECK(err, err = kernel_d_softplus.setArg(3, *(PD->fpga_ptr)));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_softplus, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_SOFTPLUS, 1);
}

// -----------------------------------------------------------------
// softsign
//
void fpga_cpuemu_softsign(Tensor *A, Tensor *B){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_softsign(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_softsign(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SOFTSIGN, 0);
#ifndef K_ENABLED_SOFTSIGN
  fpga_cpuemu_softsign(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_softsign.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_softsign.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_softsign.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_softsign, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SOFTSIGN, 1);
}

// -----------------------------------------------------------------
// d_softsign
//
void fpga_cpuemu_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_softsign(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_SOFTSIGN, 0);
#ifndef K_ENABLED_D_SOFTSIGN
  fpga_cpuemu_d_softsign(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_softsign.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softsign.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softsign.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softsign.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_softsign, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_SOFTSIGN, 1);
}

// -----------------------------------------------------------------
// linear
//
void fpga_cpuemu_linear(Tensor *A, Tensor *B, float param){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_linear(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_linear(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_LINEAR, 0);
#ifndef K_ENABLED_LINEAR
  fpga_cpuemu_linear(A, B, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_linear.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_linear.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_linear.setArg(2, param));
  OCL_CHECK(err, err = kernel_linear.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_linear, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_LINEAR, 1);
}

// -----------------------------------------------------------------
// d_linear
//
void fpga_cpuemu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_linear(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_LINEAR, 0);
#ifndef K_ENABLED_D_LINEAR
  fpga_cpuemu_d_linear(D, I, PD, param);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_linear.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_linear.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_linear.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_linear.setArg(3, param));
  OCL_CHECK(err, err = kernel_d_linear.setArg(4, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_linear, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_LINEAR, 1);
}

// -----------------------------------------------------------------
// sigmoid
//
void fpga_cpuemu_sigmoid(Tensor *A, Tensor *B){
  fpga_copy_from_fpga(A, A->ptr);
  _profile_fpga_tensor(A);
  cpu_sigmoid(A, B);
  fpga_copy_to_fpga(B->ptr, B);
  _profile_fpga_tensor(B);
  _profile_fpga_tensor_print(B);
}

void fpga_sigmoid(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SIGMOID, 0);
#ifndef K_ENABLED_SIGMOID
  fpga_cpuemu_sigmoid(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sigmoid.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sigmoid.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sigmoid.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sigmoid, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SIGMOID, 1);
}

// -----------------------------------------------------------------
// d_sigmoid
//
void fpga_cpuemu_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_sigmoid(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_SIGMOID, 0);
#ifndef K_ENABLED_D_SIGMOID
  fpga_cpuemu_d_sigmoid(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_sigmoid.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_sigmoid.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_sigmoid.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_sigmoid.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_sigmoid, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_SIGMOID, 1);
}

// -----------------------------------------------------------------
// hard_sigmoid
//
void fpga_cpuemu_hard_sigmoid(Tensor *A, Tensor *B){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_hard_sigmoid(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_hard_sigmoid(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_HARD_SIGMOID, 0);
#ifndef K_ENABLED_HARD_SIGMOID
  fpga_cpuemu_hard_sigmoid(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_hard_sigmoid.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_hard_sigmoid.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_hard_sigmoid.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_hard_sigmoid, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_HARD_SIGMOID, 1);
}

// -----------------------------------------------------------------
// d_hard_sigmoid
//
void fpga_cpuemu_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_hard_sigmoid(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_HARD_SIGMOID, 0);
#ifndef K_ENABLED_D_HARD_SIGMOID
  fpga_cpuemu_d_hard_sigmoid(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_hard_sigmoid, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_HARD_SIGMOID, 1);
}

// -----------------------------------------------------------------
// d_exp
//
void fpga_cpuemu_d_exp(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_exp(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_exp(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_EXP, 0);
#ifndef K_ENABLED_D_EXP
  fpga_cpuemu_d_exp(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_exp.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_exp.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_exp.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_exp.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_exp, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_EXP, 1);
}

// -----------------------------------------------------------------
// d_tanh
//
void fpga_cpuemu_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_tanh(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_TANH, 0);
#ifndef K_ENABLED_D_TANH
  fpga_cpuemu_d_tanh(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_tanh.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_tanh.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_tanh.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_tanh.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_tanh, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_TANH, 1);
}

// -----------------------------------------------------------------
// softmax
//
void fpga_cpuemu_softmax(Tensor *A, Tensor *B){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_softmax(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_softmax(Tensor *A, Tensor *B) {
  _debug_fpga_funcs("softmax");
  _profile_fpga(_FPGA_SOFTMAX, 0);
#ifndef K_ENABLED_SOFTMAX
  fpga_cpuemu_softmax(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_softmax.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_softmax.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_softmax.setArg(2, (int)A->shape[0]));
  OCL_CHECK(err, err = kernel_softmax.setArg(3, (int)A->shape[1]));
  OCL_CHECK(err, err = kernel_softmax.setArg(4, (int)B->shape[1]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_softmax, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SOFTMAX, 1);
  _profile_fpga_tensor(A);
  _profile_fpga_tensor(B);
}

// -----------------------------------------------------------------
// full softmax
// softmax
//
void fpga_cpuemu_full_softmax(Tensor *A, Tensor *B, int axis, bool stable) {

  fpga_copy_from_fpga(A, A->ptr, 0);
  fpga_data_type *ptrA = (fpga_data_type *)A->ptr;
  fpga_data_type *ptrB = (fpga_data_type *)B->ptr;

  int chuck_size = A->shape[axis];
  int n_samples = A->size/chuck_size;
  int inner_stride = A->stride[axis];
  int sample_stride = chuck_size*A->stride[axis];
  int k_stride = (chuck_size-1)*A->stride[axis];

  #pragma omp parallel for
  for(int si=0; si<n_samples; si++) {  // n chucks
    int start_b = si % inner_stride + si/inner_stride * sample_stride;
    int end_b = start_b + k_stride;

    // Case: Shape=(100, 3, 5, 5); Stride=(75, 25, 5, 1)
    // Action: 1) Remove dimensions (virtually), 2) Jump from your axis stride
    // Example: 1) axis=1 => 0, 25, 75...   |   2) axis=2 => 0, 5, 10, 15,...
    // for(int i=0; i<batch_stride; i+=A->stride[axis]){ ... }

    // Numerical stability (opt.)
    // stable => first value, no stable => 0.0f
    fpga_data_type max_value = (fpga_data_type)CPU_LOWEST_FLOAT;
    if (stable) {
      for (int i = start_b; i <= end_b; i += inner_stride) {
        if (ptrA[i] > max_value) { max_value = ptrA[i]; }
      }
    }

    // Numerator
    fpga_data_type denominator = (fpga_data_type)CPU_EPS_FLOAT;
    for (int i = start_b; i <= end_b; i += inner_stride) {
      fpga_data_type value = ::expf(ptrA[i] - max_value);  // Highest number should be zero
      ptrB[i] = value;
      denominator += value;
    }

    // Softmax
    for (int i = start_b; i <= end_b; i += inner_stride) {
      ptrB[i] /= denominator;
    }
  }

  fpga_copy_to_fpga(B->ptr, B, 0);
}

void fpga_full_softmax(Tensor *A, Tensor *B, int axis, bool stable) {
  _debug_fpga_funcs("(full)softmax");
  _profile_fpga(_FPGA_SOFTMAX, 0);
#ifndef K_ENABLED_SOFTMAX
  fpga_cpuemu_full_softmax(A, B, axis, stable);
#else
  printf("kernel full softmax not implemented yet\n");
  exit(1);
#endif
  _profile_fpga(_FPGA_SOFTMAX, 1);
  _profile_fpga_tensor(A);
  _profile_fpga_tensor(B);
}


// -----------------------------------------------------------------
// d_softmax
//
void fpga_cpuemu_d_softmax(Tensor *D, Tensor *I, Tensor *PD){
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_softmax(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_softmax(Tensor *D, Tensor *I, Tensor *PD) {
 _profile_fpga(_FPGA_D_SOFTMAX, 0);

#ifndef K_ENABLED_D_SOFTMAX
  fpga_cpuemu_d_softmax(D, I, PD);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_softmax.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softmax.setArg(1, *(I->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softmax.setArg(2, *(PD->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_softmax.setArg(3, (long int)D->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_d_softmax, NULL, &event));
  q.finish();
#endif

  _profile_fpga(_FPGA_D_SOFTMAX, 1);
}

#endif
