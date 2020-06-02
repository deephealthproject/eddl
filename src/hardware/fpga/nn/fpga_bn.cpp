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


// BN
void fpga_permute_channels_last(Tensor *A,Tensor *B)
{
  _profile_fpga(_FPGA_PERMUTE_CHANELS_LAST, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_PERMUTE_CHANELS_LAST, 1);

}

void fpga_permute_channels_first(Tensor *A,Tensor *B)
{
  _profile_fpga(_FPGA_PERMUTE_CHANELS_FIRST, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_PERMUTE_CHANELS_FIRST, 1);

}

void fpga_permute_batch_last(Tensor *A,Tensor *B)
{
  _profile_fpga(_FPGA_PERMUTE_BATCH_LAST, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_PERMUTE_BATCH_LAST, 1);
}

void fpga_permute_batch_first(Tensor *A,Tensor *B)
{
  _profile_fpga(_FPGA_PERMUTE_BATCH_FIRST, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_PERMUTE_BATCH_FIRST, 1);
}
