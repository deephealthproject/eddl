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

int fpga_accuracy(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_ACCURACY, 0);
  printf("fpga_accuracy not yet implemented\n"); exit(1);
  _profile_fpga(_FPGA_ACCURACY, 1);
  return 0;
}
