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

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_conv2D      = 1;
char fpga_set_cpuemu_conv2D_grad = 1;
char fpga_set_cpuemu_conv2D_back = 1;

// -----------------------------------------------------------------
// conv2D
//
void fpga_cpuemu_conv2D(ConvolDescriptor *D) {
 /* int Ksize = D->K->size * sizeof(float);
  int Isize = D->ptrI->size * sizeof(float);
  if (D->K->ptr == NULL) D->K->ptr = (float *)malloc(Ksize);
  if (D->ptrI->ptr == NULL) D->ptrI->ptr = (float *)malloc(Isize);
  fpga_copy_from_fpga(D->K, D->K->ptr);
  fpga_copy_from_fpga(D->ptrI, D->ptrI->ptr);
  cpu_conv2D(D);
  fpga_copy_to_fpga(->ptr, B);*/
  printf("fpga_cpuemu_conv2D not implemented yet\n");
  exit(1);
}

void fpga_conv2D(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D, 0);
  if (fpga_set_cpuemu_conv2D == 1) {
    fpga_cpuemu_conv2D(D);
  } else {
      printf("fpga_conv2D not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_CONV2D, 1);
}

// -----------------------------------------------------------------
// conv2D_grad
//
void fpga_cpuemu_conv2D_grad(ConvolDescriptor *D) {
    printf("fpga_cpuemu_conv2D_grad not implemented yet\n");
    exit(1);
}

void fpga_conv2D_grad(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D_GRAD, 0);
  if (fpga_set_cpuemu_conv2D_grad == 1) {
    fpga_cpuemu_conv2D_grad(D);
  } else {
      printf("fpga_conv2D_grad not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_CONV2D_GRAD, 1);
}

// -----------------------------------------------------------------
// conv2D_back
//
void fpga_cpuemu_conv2D_back(ConvolDescriptor *D) {
    printf("fpga_cpuemu_conv2D_back not implemented yet\n");
    exit(1);
}

void fpga_conv2D_back(ConvolDescriptor *D)
{
  _profile_fpga(_FPGA_CONV2D_BACK, 0);
  if (fpga_set_cpuemu_conv2D_back == 1) {
    fpga_cpuemu_conv2D_back(D);
  } else {
      printf("fpga_conv2D_back not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_CONV2D_BACK, 1);
}
