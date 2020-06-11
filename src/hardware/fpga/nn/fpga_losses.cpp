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
#include "eddl/hardware/cpu/nn/cpu_nn.h"
#include "eddl/hardware/fpga/fpga_hw.h"

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_cent      = 1;

// -----------------------------------------------------------------
// cent
//
void fpga_cpuemu_cent(Tensor *A, Tensor *B, Tensor *C) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_cent(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_cent(Tensor *A, Tensor *B, Tensor *C){
  _profile_fpga(_FPGA_CENT, 0);

  if (fpga_set_cpuemu_cent == 1) {
    fpga_cpuemu_cent(A, B, C);
  } else {
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_cent.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cent.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cent.setArg(2, (C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cent.setArg(3, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_cent, NULL, &event));
    //  event.wait();
    q.finish();
  }
  _profile_fpga(_FPGA_CENT, 1);
}
