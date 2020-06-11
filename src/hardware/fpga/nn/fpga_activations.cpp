/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/



#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/fpga/fpga_hw.h"   // for buffer copies
#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_nn.h"  // for cpu emulation purposes

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_relu               = 0;
char fpga_set_cpuemu_d_relu             = 1;
char fpga_set_cpuemu_thresholded_relu   = 1;
char fpga_set_cpuemu_d_thresholded_relu = 1;
char fpga_set_cpuemu_leaky_relu         = 1;
char fpga_set_cpuemu_d_leaky_relu       = 1;
char fpga_set_cpuemu_elu                = 1;
char fpga_set_cpuemu_d_elu              = 1;
char fpga_set_cpuemu_softplus           = 1;
char fpga_set_cpuemu_d_softplus         = 1;
char fpga_set_cpuemu_softsign           = 1;
char fpga_set_cpuemu_d_softsign         = 1;
char fpga_set_cpuemu_linear             = 1;
char fpga_set_cpuemu_d_linear           = 1; 
char fpga_set_cpuemu_sigmoid            = 1;
char fpga_set_cpuemu_d_sigmoid          = 1;
char fpga_set_cpuemu_hard_sigmoid       = 1;
char fpga_set_cpuemu_d_hard_sigmoid     = 1;
char fpga_set_cpuemu_exp                = 1;
char fpga_set_cpuemu_d_exp              = 1;
char fpga_set_cpuemu_tanh               = 1;
char fpga_set_cpuemu_d_tanh             = 1;
char fpga_set_cpuemu_softmax            = 1;
char fpga_set_cpuemu_d_softmax          = 1;

// -----------------------------------------------------------------
// relu
//
void fpga_cpuemu_relu(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_relu(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_relu(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_RELU, 0);
  _profile_fpga_tensor(A);
  if (fpga_set_cpuemu_relu == 1) {
      fpga_cpuemu_relu(A, B);
  } else {
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_relu.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_relu.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_relu.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_relu, NULL, &event));
    //  event.wait();
    q.finish();
  }
  _profile_fpga_tensor(B);
  _profile_fpga(_FPGA_RELU, 1);
}

// -----------------------------------------------------------------
// d_relu
//
void fpga_cpuemu_d_relu(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_relu(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_relu(Tensor *D, Tensor *I, Tensor *PD){
 _profile_fpga(_FPGA_D_RELU, 0);
 if (fpga_set_cpuemu_d_relu == 1) {
     fpga_cpuemu_d_relu(D, I, PD);
 } else {
     printf("fpga_d_relu not implemented yet\n"); exit(1);
 }
 _profile_fpga(_FPGA_D_RELU, 1);
}

// -----------------------------------------------------------------
// thbresholded_relu
//
void fpga_cpuemu_thresholded_relu(Tensor *A, Tensor *B, float param){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_thresholded_relu(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_thresholded_relu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_THRESHOLDED_RELU, 0);
  if (fpga_set_cpuemu_thresholded_relu == 1) {
      fpga_cpuemu_thresholded_relu(A, B, param);
  } else {
      printf("fpga_thresholded_relu not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_THRESHOLDED_RELU, 1);
}

// -----------------------------------------------------------------
// d_thresholded_relu
//
void fpga_cpuemu_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_thresholded_relu(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_THRESHOLDED_RELU, 0);
  if (fpga_set_cpuemu_d_thresholded_relu == 1) {
      fpga_cpuemu_d_thresholded_relu(D, I, PD, param);
  } else {
      printf("fpga_d_thresholded_relu not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_THRESHOLDED_RELU, 1);
}

// -----------------------------------------------------------------
// leaky_relu
//
void fpga_cpuemu_leaky_relu(Tensor *A, Tensor *B, float param){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_leaky_relu(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_leaky_relu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_LEAKY_RELU, 0);
  if (fpga_set_cpuemu_leaky_relu == 1) {
      fpga_cpuemu_leaky_relu(A, B, param);
  } else {
      printf("fpga_leaky_relu not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_LEAKY_RELU, 1);
}

// -----------------------------------------------------------------
// d_leaky_relu
//
void fpga_cpuemu_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_leaky_relu(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
  _profile_fpga(_FPGA_D_LEAKY_RELU, 0);
  if (fpga_set_cpuemu_d_leaky_relu == 1) {
      fpga_cpuemu_d_leaky_relu(D, I, PD, param);
  } else {
      printf("fpga_leaky_relu not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_LEAKY_RELU, 1);
}

// -----------------------------------------------------------------
// elu
//
void fpga_cpuemu_elu(Tensor *A, Tensor *B, float param){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_elu(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_elu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_ELU, 0);
  if (fpga_set_cpuemu_elu == 1) {
      fpga_cpuemu_elu(A, B, param);
  } else {
      printf("fpga_elu not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_ELU, 1);
}

// -----------------------------------------------------------------
// d_elu
//
void fpga_cpuemu_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_elu(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_ELU, 0);
  if (fpga_set_cpuemu_d_elu == 1) {
      fpga_cpuemu_d_elu(D, I, PD, param);
  } else {
      printf("fpga_d_elu not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_ELU, 1);
}

// -----------------------------------------------------------------
// softplus
//
void fpga_cpuemu_softplus(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_softplus(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_softplus(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_SOFTPLUS, 0);
    if (fpga_set_cpuemu_softplus == 1) {
        fpga_cpuemu_softplus(A, B);
    } else {
        printf("fpga_softplus not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SOFTPLUS, 1);
}

// -----------------------------------------------------------------
// d_softplus
//
void fpga_cpuemu_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_softplus(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
    _profile_fpga(_FPGA_D_SOFTPLUS, 0);
    if (fpga_set_cpuemu_d_softplus == 1) {
        fpga_cpuemu_d_softplus(D, I, PD);
    } else {
        printf("fpga_d_softplus not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_D_SOFTPLUS, 1);
}

// -----------------------------------------------------------------
// softsign
//
void fpga_cpuemu_softsign(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_softsign(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_softsign(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_SOFTSIGN, 0);
    if (fpga_set_cpuemu_softsign == 1) {
        fpga_cpuemu_softsign(A, B);
    } else {
        printf("fpga_softsign not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SOFTSIGN, 1);
}

void fpga_cpuemu_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_softsign(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

// -----------------------------------------------------------------
// d_softsign
//
void fpga_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
    _profile_fpga(_FPGA_D_SOFTSIGN, 0);
    if (fpga_set_cpuemu_d_softsign == 1) {
        fpga_cpuemu_d_softsign(D, I, PD);
    } else {
        printf("fpga_d_softsign not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_D_SOFTSIGN, 1);
}

// -----------------------------------------------------------------
// linear
//
void fpga_cpuemu_linear(Tensor *A, Tensor *B, float param){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_linear(A, B, param);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_linear(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_LINEAR, 0);
  if (fpga_set_cpuemu_linear == 1) {
      fpga_cpuemu_linear(A, B, param);
  } else {
      printf("fpga_linear not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_LINEAR, 1);
}

// -----------------------------------------------------------------
// d_linear
//
void fpga_cpuemu_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_linear(D, I, PD, param);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_LINEAR, 0);
  if (fpga_set_cpuemu_d_linear == 1) {
      fpga_cpuemu_d_linear(D, I, PD, param);
  } else {
      printf("fpga_d_linear not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_LINEAR, 1);
}

// -----------------------------------------------------------------
// sigmoid
//
void fpga_cpuemu_sigmoid(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sigmoid(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sigmoid(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SIGMOID, 0);
  if (fpga_set_cpuemu_sigmoid == 1) {
      fpga_cpuemu_sigmoid(A, B);
  } else {
      printf("fpga_sigmoid not implemented yet\n"); exit(1);
  };
  _profile_fpga(_FPGA_SIGMOID, 1);
}

// -----------------------------------------------------------------
// d_sigmoid
//
void fpga_cpuemu_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_sigmoid(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_SIGMOID, 0);
  if (fpga_set_cpuemu_d_sigmoid == 1) {
      fpga_cpuemu_d_sigmoid(D, I, PD);
  } else {
      printf("fpga_d_sigmoid not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_SIGMOID, 1);
}

// -----------------------------------------------------------------
// hard_sigmoid
//
void fpga_cpuemu_hard_sigmoid(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_hard_sigmoid(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_hard_sigmoid(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_HARD_SIGMOID, 0);
  if (fpga_set_cpuemu_hard_sigmoid == 1) {
      fpga_cpuemu_hard_sigmoid(A, B);
  } else {
      printf("fpga_hard_sigmoid not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_HARD_SIGMOID, 1);
}

// -----------------------------------------------------------------
// d_hard_sigmoid
//
void fpga_cpuemu_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_hard_sigmoid(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_HARD_SIGMOID, 0);
  if (fpga_set_cpuemu_d_hard_sigmoid == 1) {
      fpga_cpuemu_d_hard_sigmoid(D, I, PD);
  } else {
      printf("fpga_d_hard_sigmoid not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_HARD_SIGMOID, 1);
}

// -----------------------------------------------------------------
// exp
//
void fpga_cpuemu_exp(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_exp(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_exp(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_EXP, 0);
  if (fpga_set_cpuemu_exp == 1) {
      fpga_cpuemu_exp(A, B);
  } else {
      printf("fpga_exp not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_EXP, 1);
}

// -----------------------------------------------------------------
// d_exp
//
void fpga_cpuemu_d_exp(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_exp(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_exp(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_EXP, 0);
  if (fpga_set_cpuemu_d_exp == 1) {
      fpga_cpuemu_d_exp(D, I, PD);
  } else {
      printf("fpga_d_exp not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_EXP, 1);
}

// -----------------------------------------------------------------
// tanh
//
void fpga_cpuemu_tanh(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_tanh(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_tanh(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_TANH, 0);
  if (fpga_set_cpuemu_tanh == 1) {
      fpga_cpuemu_tanh(A, B);
  } else {
      printf("fpga_tanh not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_TANH, 1);
}

// -----------------------------------------------------------------
// d_tanh
//
void fpga_cpuemu_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_tanh(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_TANH, 0);
  if (fpga_set_cpuemu_d_tanh == 1) {
      fpga_cpuemu_d_tanh(D, I, PD);
  } else {
      printf("fpga_d_tanh not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_D_TANH, 1);
}

// -----------------------------------------------------------------
// softmax
//
void fpga_cpuemu_softmax(Tensor *A, Tensor *B){
  int Asize = A->size * sizeof(float);
  int Bsize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_softmax(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_softmax(Tensor *A, Tensor *B) {
  _profile_fpga(_FPGA_SOFTMAX, 0);
  _profile_fpga_tensor(A);
  _profile_fpga_tensor(B);
  if (fpga_set_cpuemu_softmax == 1) {
      fpga_cpuemu_softmax(A, B);
  } else {
      // printf("fpga_softmax not implemented yet\n"); exit(1);
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_softmax.setArg(0, (A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_softmax.setArg(1, (B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_softmax.setArg(2, (int)A->shape[0]));
      OCL_CHECK(err, err = kernel_softmax.setArg(3, (int)A->shape[1]));
      OCL_CHECK(err, err = kernel_softmax.setArg(4, (int)B->shape[1]));

      OCL_CHECK(err, err = q.enqueueTask(kernel_softmax, NULL, &event));
      //  event.wait();
      q.finish();

  }
  _profile_fpga(_FPGA_SOFTMAX, 1);
}

// -----------------------------------------------------------------
// d_softmax
//
void fpga_cpuemu_d_softmax(Tensor *D, Tensor *I, Tensor *PD){
  int Dsize = D->size * sizeof(float);
  int Isize = I->size * sizeof(float);
  int PDsize = PD->size * sizeof(float);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  if (I->ptr == NULL) I->ptr = (float *)malloc(Isize);
  if (PD->ptr == NULL) PD->ptr = (float *)malloc(PDsize);
  fpga_copy_from_fpga(D, D->ptr);
  fpga_copy_from_fpga(I, I->ptr);
  cpu_d_softmax(D, I, PD);
  fpga_copy_to_fpga(PD->ptr, PD);
}

void fpga_d_softmax(Tensor *D, Tensor *I, Tensor *PD) {
    _profile_fpga(_FPGA_D_SOFTMAX, 0);
  PD->tsem->lock();
  if (fpga_set_cpuemu_d_softmax == 1) {
      fpga_cpuemu_d_softmax(D, I, PD);
  } else {
      printf("fpga_d_softmax not implemented yet\n"); exit(1);
  }
  PD->tsem->unlock();
  _profile_fpga(_FPGA_D_SOFTMAX, 1);
}
