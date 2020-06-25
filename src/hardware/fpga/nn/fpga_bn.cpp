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

#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"

// -----------------------------------------------------------------
// permute_channels_last
//
void fpga_cpuemu_permute_channels_last(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_permute_channels_last(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_permute_channels_last(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_PERMUTE_CHANELS_LAST, 0);
#ifndef K_ENABLED_PERMUTE_CHANNELS_LAST
  fpga_cpuemu_permute_channels_last(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_permute_channels_last.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_channels_last.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_channels_last.setArg(2, (int)A->shape[0]));
  OCL_CHECK(err, err = kernel_permute_channels_last.setArg(3, (int)A->shape[1]));
  OCL_CHECK(err, err = kernel_permute_channels_last.setArg(4, (int)A->shape[2]));
  OCL_CHECK(err, err = kernel_permute_channels_last.setArg(5, (int)A->shape[3]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_permute_channels_last, NULL, &event));
  q.finish();
#endif
_profile_fpga(_FPGA_PERMUTE_CHANELS_LAST, 1);

}

// -----------------------------------------------------------------
// permute_channels_first
//
void fpga_cpuemu_permute_channels_first(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_permute_channels_first(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_permute_channels_first(Tensor *A,Tensor *B){
  _profile_fpga(_FPGA_PERMUTE_CHANELS_FIRST, 0);
#ifndef K_ENABLED_PERMUTE_CHANNELS_FIRST
  fpga_cpuemu_permute_channels_first(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_permute_channels_first.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_channels_first.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_channels_first.setArg(2, (int)B->shape[0]));
  OCL_CHECK(err, err = kernel_permute_channels_first.setArg(3, (int)B->shape[1]));
  OCL_CHECK(err, err = kernel_permute_channels_first.setArg(4, (int)B->shape[2]));
  OCL_CHECK(err, err = kernel_permute_channels_first.setArg(5, (int)B->shape[3]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_permute_channels_first, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_PERMUTE_CHANELS_FIRST, 1);

}

// -----------------------------------------------------------------
// permute_batch_last
//
void fpga_cpuemu_permute_batch_last(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_permute_batch_last(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_permute_batch_last(Tensor *A,Tensor *B){
  _profile_fpga(_FPGA_PERMUTE_BATCH_LAST, 0);
#ifndef K_ENABLED_PERMUTE_BATCH_LAST
  fpga_cpuemu_permute_batch_last(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_permute_batch_last.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_batch_last.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_batch_last.setArg(2, (int)A->shape[0]));
  OCL_CHECK(err, err = kernel_permute_batch_last.setArg(3, (int)A->shape[1]));
  OCL_CHECK(err, err = kernel_permute_batch_last.setArg(4, (int)A->shape[2]));
  OCL_CHECK(err, err = kernel_permute_batch_last.setArg(5, (int)A->shape[3]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_permute_batch_last, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_PERMUTE_BATCH_LAST, 1);
}

// -----------------------------------------------------------------
// permute_batch_first
//
void fpga_cpuemu_permute_batch_first(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_permute_batch_first(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_permute_batch_first(Tensor *A,Tensor *B){
  _profile_fpga(_FPGA_PERMUTE_BATCH_FIRST, 0);
#ifndef K_ENABLED_PERMUTE_BATCH_FIRST
  fpga_cpuemu_permute_batch_first(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_permute_batch_first.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_batch_first.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_permute_batch_first.setArg(2, (int)B->shape[0]));
  OCL_CHECK(err, err = kernel_permute_batch_first.setArg(3, (int)B->shape[1]));
  OCL_CHECK(err, err = kernel_permute_batch_first.setArg(4, (int)B->shape[2]));
  OCL_CHECK(err, err = kernel_permute_batch_first.setArg(5, (int)B->shape[3]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_permute_batch_first, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_PERMUTE_BATCH_FIRST, 1);
}
