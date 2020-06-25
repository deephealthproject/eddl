/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include "eddl/hardware/cpu/cpu_tensor.h"
#include <limits>

#ifdef cFPGA
#include "eddl/hardware/fpga/fpga_hw.h"
#endif

// -------------------------------------------------------------------------
// where
//
void fpga_cpuemu_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  fpga_copy_from_fpga(condition, condition->ptr);
  cpu_where(condition, A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_where(Tensor *condition, Tensor *A, Tensor *B, Tensor *C){
#ifndef K_ENABLED_WHERE
  fpga_cpuemu_where(condition, A, B, C);
#else
  printf("fpga_where not implemented yet\n");
  exit(1);
#endif
}
