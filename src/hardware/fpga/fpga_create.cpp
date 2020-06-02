/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/



#include "eddl/hardware/fpga/fpga_hw.h"

void fpga_range(Tensor *A, float min, float step){
    _profile_fpga(_FPGA_RANGE, 0);
    printf("fpga_range not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RANGE, 1);
}

void fpga_eye(Tensor *A, int offset){
    _profile_fpga(_FPGA_EYE, 0);
    printf("fpga_eye not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_EYE, 1);
}
