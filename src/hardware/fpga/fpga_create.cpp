/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/hardware/fpga/fpga_hw.h"

#include "eddl/hardware/cpu/cpu_tensor.h"

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_range_     = 1;
char fpga_set_cpuemu_eye_       = 1;

// -----------------------------------------------------------------
// range
//
void fpga_cpuemu_range(Tensor *A, float min, float step) {
  cpu_range(A, min, step);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_range(Tensor *A, float min, float step){
  _profile_fpga(_FPGA_RANGE, 0);
#ifndef K_ENABLED_RANGE
  fpga_cpuemu_range(A, min, step);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_range.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_range.setArg(1, min));
  OCL_CHECK(err, err = kernel_range.setArg(2, step));
  OCL_CHECK(err, err = kernel_range.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_deselect, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_RANGE, 1);
}

// -----------------------------------------------------------------
// eye
//
void fpga_cpuemu_eye(Tensor *A, int offset) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  cpu_eye(A, offset);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_eye(Tensor *A, int offset){
  _profile_fpga(_FPGA_EYE, 0);
#ifndef K_ENABLED_EYE
  fpga_cpuemu_eye(A, offset);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_range.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_range.setArg(1, offset));
  OCL_CHECK(err, err = kernel_range.setArg(2, (long int)A->size));
  OCL_CHECK(err, err = kernel_range.setArg(3, (int)A->shape[0]));
  OCL_CHECK(err, err = kernel_range.setArg(4, (int)A->shape[1]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_range, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_EYE, 1);
}

// -----------------------------------------------------------------
// diag
//
void fpga_cpuemu_diag(Tensor *A, Tensor *B, int k) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_diag(A, B, k);
  fpga_copy_to_fpga(B->ptr, B);
}

void cpu_diag(Tensor *A, Tensor *B, int k){
#ifndef K_ENABLED_DIAG
  fpga_cpuemu_diag(A, B, k);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_diag.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_diag.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_diag.setArg(2, (long int)A->size));
  OCL_CHECK(err, err = kernel_diag.setArg(3, (int)k);

  OCL_CHECK(err, err = q.enqueueTask(kernel_diag, NULL, &event));
  q.finish();
#endif
}
