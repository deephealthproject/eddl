/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_repeat_nn       = 1;
char fpga_set_cpuemu_d_repeat_nn     = 1;


// -----------------------------------------------------------------
// repeat_nn
//
void fpga_cpuemu_repeat_nn(Tensor *A, Tensor *B, vector<int> size) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_repeat_nn(A, B, size);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
    _profile_fpga(_FPGA_REPEAT_NN, 0);
    if (fpga_set_cpuemu_repeat_nn == 1) {
        fpga_cpuemu_repeat_nn(A, B, size);
    } else {
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_repeat_nn.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_repeat_nn.setArg(1, *(B->fpga_ptr)));
	printf("error, parameter fpga\n");
//        OCL_CHECK(err, err = kernel_repeat_nn.setArg(2, (int)size->fpga_ptr));

        OCL_CHECK(err, err = q.enqueueTask(kernel_repeat_nn, NULL, &event));
        q.finish();
    }
    _profile_fpga(_FPGA_REPEAT_NN, 1);
}

// -----------------------------------------------------------------
// d_repeat_nn
//
void fpga_cpuemu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
  int Asize = A->size * sizeof(float);
  int Dsize = D->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (D->ptr == NULL) D->ptr = (float *)malloc(Dsize);
  fpga_copy_from_fpga(D, D->ptr);
  cpu_repeat_nn(D, A, size);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    _profile_fpga(_FPGA_D_REPEAT_NN, 0);
    if (fpga_set_cpuemu_d_repeat_nn == 1) {
        fpga_cpuemu_d_repeat_nn(D, A, size);
    } else {
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(0, *(D->fpga_ptr)));
        OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(1, *(A->fpga_ptr)));
//        OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(2, (int)size->fpga_ptr));
        printf("error, parameter fpga\n");
        OCL_CHECK(err, err = q.enqueueTask(kernel_d_repeat_nn, NULL, &event));
        q.finish();
    }
    _profile_fpga(_FPGA_D_REPEAT_NN, 1);
}

// -------------------------------------------------------------------
// select_nn
//
void fpga_cpuemu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  printf("cpuemu_select_nn not implemented yet\n");
  exit(1);
}

void fpga_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SELECT_NN
  fpga_cpuemu_select_nn(A, B, sd);
#else 
  printf("fpga_select_nn not implemented yet\n");
  exit(1);
#endif
}

// ------------------------------------------------------------------
// select_back_nn
//
void fpga_cpuemu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  printf("cpuemu_select_back_nn not implemented yet\n");
  exit(1);
}

void fpga_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SELECT_BACK_NN
  fpga_cpuemu_select_back_nn(A, B, sd);
#else
  printf("fpga_select_back_nn not implemented yet\n");
  exit(1);
#endif
}

// -----------------------------------------------------------------
// set_select_nn
//
void fpga_cpuemu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  printf("cpuemu_set_select_nn not implemented yet\n");
  exit(1);
}

void fpga_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SET_SELECT_NN
  fpga_cpuemu_set_select_nn(A, B, sd);
#else
  printf("fpga_set_select_nn not implemented yet\n");
  exit(1);
#endif
}

// -----------------------------------------------------------------
// set_select_back_nn
//
void fpga_cpuemu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  printf("cpuemu_set_select_back_nn not implemented yet\n");
  exit(1);
}

void fpga_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SET_SELECT_BACK_NN
  fpga_cpuemu_set_select_back_nn(A, B, sd);
#else
  printf("fpga_set_select_back_nn not implemented yet\n");
  exit(1);
#endif
}

