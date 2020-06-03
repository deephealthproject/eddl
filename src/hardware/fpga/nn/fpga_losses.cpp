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

extern cl::Kernel kernel_cent;
extern cl::CommandQueue q;

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_cent      = 1;

// -----------------------------------------------------------------
// cent
//
void fpga_cpuemu_cent(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_cent not implemented yet\n");
    exit(1);
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
    OCL_CHECK(err, err = kernel_cent.setArg(3, A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_cent, NULL, &event));
    //  event.wait();
    q.finish();
  }
  _profile_fpga(_FPGA_CENT, 1);
}
