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
char fpga_set_cpuemu_relu               = 1;
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

    OCL_CHECK(err, err = kernel_relu.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_relu.setArg(1, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_relu.setArg(2, A->size));

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
     cl_int err;
     cl::Event event;

     OCL_CHECK(err, err = kernel_d_relu.setArg(0, *(D->fpga_ptr)));
     OCL_CHECK(err, err = kernel_d_relu.setArg(1, *(I->fpga_ptr)));
     OCL_CHECK(err, err = kernel_d_relu.setArg(2, *(PD->fpga_ptr)));
     OCL_CHECK(err, err = kernel_d_relu.setArg(3, (long int)D->size));

     OCL_CHECK(err, err = q.enqueueTask(kernel_d_relu, NULL, &event));
     q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_thresholded_relu.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_thresholded_relu.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_thresholded_relu.setArg(2, (long int)A->size));
      OCL_CHECK(err, err = kernel_thresholded_relu.setArg(3, param));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_relu, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(3, (long int)D->size));
      OCL_CHECK(err, err = kernel_d_thresholded_relu.setArg(4, param));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_thresholded_relu, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_leaky_relu.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_leaky_relu.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_leaky_relu.setArg(2, (long int)A->size));
      OCL_CHECK(err, err = kernel_leaky_relu.setArg(4, param));

      OCL_CHECK(err, err = q.enqueueTask(kernel_leaky_relu, NULL, &event));
      q.finish();
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

void fpga_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_LEAKY_RELU, 0);
  if (fpga_set_cpuemu_d_leaky_relu == 1) {
      fpga_cpuemu_d_leaky_relu(D, I, PD, param);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(3, (long int)D->size));
      OCL_CHECK(err, err = kernel_d_leaky_relu.setArg(4, param));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_leaky_relu, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_elu.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_elu.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_elu.setArg(3, (long int)A->size));
      OCL_CHECK(err, err = kernel_elu.setArg(4, param));

      OCL_CHECK(err, err = q.enqueueTask(kernel_elu, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_elu.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_elu.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_elu.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_elu.setArg(3, (long int)D->size));
      OCL_CHECK(err, err = kernel_d_elu.setArg(4, param));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_elu, NULL, &event));
      q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_softplus.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_softplus.setArg(1, *(B->fpga_ptr)));
        OCL_CHECK(err, err = kernel_softplus.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_softplus, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_d_softplus.setArg(0, *(D->fpga_ptr)));
        OCL_CHECK(err, err = kernel_d_softplus.setArg(1, *(I->fpga_ptr)));
        OCL_CHECK(err, err = kernel_d_softplus.setArg(2, (long int)D->size));
        OCL_CHECK(err, err = kernel_d_softplus.setArg(3, *(PD->fpga_ptr)));

        OCL_CHECK(err, err = q.enqueueTask(kernel_d_softplus, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_softsign.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_softsign.setArg(1, *(B->fpga_ptr)));
        OCL_CHECK(err, err = kernel_softsign.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_softsign, NULL, &event));
        q.finish();
    }
    _profile_fpga(_FPGA_SOFTSIGN, 1);
}

// -----------------------------------------------------------------
// d_softsign
//
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

void fpga_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
    _profile_fpga(_FPGA_D_SOFTSIGN, 0);
    if (fpga_set_cpuemu_d_softsign == 1) {
        fpga_cpuemu_d_softsign(D, I, PD);
    } else {
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_d_softsign.setArg(0, *(D->fpga_ptr)));
        OCL_CHECK(err, err = kernel_d_softsign.setArg(1, *(I->fpga_ptr)));
        OCL_CHECK(err, err = kernel_d_softsign.setArg(2, *(PD->fpga_ptr)));
        OCL_CHECK(err, err = kernel_d_softsign.setArg(3, (long int)D->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_d_softsign, NULL, &event));
        q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_linear.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_linear.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_linear.setArg(2, param));
      OCL_CHECK(err, err = kernel_linear.setArg(3, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_linear, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_linear.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_linear.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_linear.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_linear.setArg(3, param));
      OCL_CHECK(err, err = kernel_d_linear.setArg(4, (long int)D->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_linear, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_sigmoid.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_sigmoid.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_sigmoid.setArg(2, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_sigmoid, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_sigmoid.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_sigmoid.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_sigmoid.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_sigmoid.setArg(3, (long int)D->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_sigmoid, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_hard_sigmoid.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_hard_sigmoid.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_hard_sigmoid.setArg(2, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_hard_sigmoid, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_hard_sigmoid.setArg(3, (long int)D->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_hard_sigmoid, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_exp.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_exp.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_exp.setArg(2, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_exp, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_exp.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_exp.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_exp.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_exp.setArg(3, (long int)D->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_exp, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_tanh.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_tanh.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_tanh.setArg(2, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_tanh, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_tanh.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_tanh.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_tanh.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_tanh.setArg(3, (long int)D->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_tanh, NULL, &event));
      q.finish();
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_softmax.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_softmax.setArg(1, *(B->fpga_ptr)));
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
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_d_softmax.setArg(0, *(D->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_softmax.setArg(1, *(I->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_softmax.setArg(2, *(PD->fpga_ptr)));
      OCL_CHECK(err, err = kernel_d_softmax.setArg(3, (long int)D->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_d_softmax, NULL, &event));
      q.finish();
  }
  PD->tsem->unlock();
  _profile_fpga(_FPGA_D_SOFTMAX, 1);
}
