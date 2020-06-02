/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#include "eddl/random.h"
#include "eddl/hardware/fpga/fpga_hw.h"

void
fpga_rand_uniform(Tensor * A, float v)
{
    _profile_fpga(_FPGA_RAND_UNIFORM, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RAND_UNIFORM, 1);
}

void
fpga_rand_signed_uniform(Tensor * A, float v)
{
    _profile_fpga(_FPGA_RAND_SIGNED_UNIFORM, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RAND_SIGNED_UNIFORM, 1);
}

void
fpga_rand_binary(Tensor * A, float v)
{
    _profile_fpga(_FPGA_BINARY, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_BINARY, 1);
}

void fpga_rand_normal(Tensor * A, float m, float s, bool fast_math) {
    _profile_fpga(_FPGA_RAND_NORMAL, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RAND_NORMAL, 0);
}
