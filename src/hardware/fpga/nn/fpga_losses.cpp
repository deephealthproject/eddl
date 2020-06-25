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

#include "eddl/hardware/fpga/nn/fpga_nn.h"
#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"

// -----------------------------------------------------------------
// cent
//
void fpga_cpuemu_cent(Tensor *A, Tensor *B, Tensor *C) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_cent(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_cent(Tensor *A, Tensor *B, Tensor *C){
  _profile_fpga(_FPGA_CENT, 0);
  _profile_fpga_tensor(A);
  _profile_fpga_tensor(B);
  _profile_fpga_tensor(C);
#ifndef K_ENABLED_CENT
  fpga_cpuemu_cent(A, B, C);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_cent.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cent.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cent.setArg(2, *(C->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cent.setArg(3, A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_cent, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_CENT, 1);
}

#endif
