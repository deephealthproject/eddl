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

void fpga_conv2D(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_CONV2D, 1);
}

void fpga_conv2D_grad(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D_GRAD, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_CONV2D_GRAD, 1);
}

void fpga_conv2D_back(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D_BACK, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_CONV2D_BACK, 1);
}
