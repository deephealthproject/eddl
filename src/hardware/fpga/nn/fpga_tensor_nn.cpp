/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

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
  fpga_copy_from_fpga(A, A->ptr);
  cpu_repeat_nn(A, B, size);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
  _profile_fpga(_FPGA_REPEAT_NN, 0);
#ifndef K_ENABLED_REPEAT_NN
  fpga_cpuemu_repeat_nn(A, B, size);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_repeat_nn.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_repeat_nn.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_repeat_nn.setArg(2, B->shape[3]));
  OCL_CHECK(err, err = kernel_repeat_nn.setArg(3, B->size));
  OCL_CHECK(err, err = kernel_repeat_nn.setArg(4, size[0]));
  OCL_CHECK(err, err = kernel_repeat_nn.setArg(5, size[1]));
  OCL_CHECK(err, err = kernel_repeat_nn.setArg(6, A->shape[3]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_repeat_nn, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_REPEAT_NN, 1);
}

// -----------------------------------------------------------------
// d_repeat_nn
//
void fpga_cpuemu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
  fpga_copy_from_fpga(D, D->ptr);
  cpu_repeat_nn(D, A, size);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
  _profile_fpga(_FPGA_D_REPEAT_NN, 0);
#ifndef K_ENABLED_D_REPEAT_NN
  fpga_cpuemu_d_repeat_nn(D, A, size);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(0, *(D->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(1, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(2, D->shape[3]));
  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(3, D->size));
  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(4, size[0]));
  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(5, size[1]));
  OCL_CHECK(err, err = kernel_d_repeat_nn.setArg(6, A->shape[3]));
  OCL_CHECK(err, err = q.enqueueTask(kernel_d_repeat_nn, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_D_REPEAT_NN, 1);
}

// -------------------------------------------------------------------
// select_nn
//
void fpga_cpuemu_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  fpga_copy_from_fpga(A, A->ptr);
  //fpga_copy_memory_from_fpga(sd->fpga_ptr, sd->cpu_addresses, B->stride[0]);
  cpu_select_nn(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SELECT_NN
  fpga_cpuemu_select_nn(A, B, sd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_select_nn.setArg(0, *A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_select_nn.setArg(1, *B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_select_nn.setArg(2, B->shape[0]));
  OCL_CHECK(err, err = kernel_select_nn.setArg(3, A->stride[0]));
  OCL_CHECK(err, err = kernel_select_nn.setArg(4, B->stride[0]));
  OCL_CHECK(err, err = kernel_select_nn.setArg(5, *sd->fpga_ptr));

  OCL_CHECK(err, err = q.enqueueTask(kernel_select_nn, NULL, &event));
  q.finish();
#endif
}

// ------------------------------------------------------------------
// select_back_nn
//
void fpga_cpuemu_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(sd->fpga_ptr, sd->cpu_addresses, A->stride[0]);
  cpu_select_back_nn(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SELECT_BACK_NN
  fpga_cpuemu_select_back_nn(A, B, sd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_select_back_nn.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_select_back_nn.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_select_back_nn.setArg(2, A->shape[0]));
  OCL_CHECK(err, err = kernel_select_back_nn.setArg(3, A->stride[0]));
  OCL_CHECK(err, err = kernel_select_back_nn.setArg(4, B->stride[0]));
  OCL_CHECK(err, err = kernel_select_back_nn.setArg(5, *sd->fpga_ptr));

  OCL_CHECK(err, err = q.enqueueTask(kernel_select_back_nn, NULL, &event));
  q.finish();
#endif
}

// -----------------------------------------------------------------
// set_select_nn
//
void fpga_cpuemu_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(sd->fpga_ptr, sd->cpu_addresses, B->stride[0]);
  cpu_set_select_nn(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_set_select_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
#ifndef K_ENABLED_SET_SELECT_NN
  fpga_cpuemu_set_select_nn(A, B, sd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_set_select_nn.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_set_select_nn.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_set_select_nn.setArg(2, B->shape[0]));
  OCL_CHECK(err, err = kernel_set_select_nn.setArg(3, A->stride[0]));
  OCL_CHECK(err, err = kernel_set_select_nn.setArg(4, B->stride[0]));
  OCL_CHECK(err, err = kernel_set_select_nn.setArg(5, *sd->fpga_ptr));

  OCL_CHECK(err, err = q.enqueueTask(kernel_set_select_nn, NULL, &event));
  q.finish();
#endif
}

// -----------------------------------------------------------------
// set_select_back_nn
//
void fpga_cpuemu_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(sd->fpga_ptr, sd->cpu_addresses, B->stride[0]);
  cpu_set_select_back_nn(A, B, sd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_set_select_back_nn(Tensor *A, Tensor *B, SelDescriptor *sd){
	printf("set_select_back_nn\n");
#ifndef K_ENABLED_SET_SELECT_BACK_NN
  fpga_cpuemu_set_select_back_nn(A, B, sd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_set_select_back_nn.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_set_select_back_nn.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_set_select_back_nn.setArg(2, B->shape[0]));
  OCL_CHECK(err, err = kernel_set_select_back_nn.setArg(3, A->stride[0]));
  OCL_CHECK(err, err = kernel_set_select_back_nn.setArg(4, B->stride[0]));
  OCL_CHECK(err, err = kernel_set_select_back_nn.setArg(5, *sd->fpga_ptr));

  OCL_CHECK(err, err = q.enqueueTask(kernel_set_select_back_nn, NULL, &event));
  q.finish();
#endif
}

#endif
