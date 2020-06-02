/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#include "eddl/hardware/fpga/nn/fpga_nn.h"

void fpga_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
    _profile_fpga(_FPGA_REPEAT_NN, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_REPEAT_NN, 1);
}

void fpga_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    _profile_fpga(_FPGA_D_REPEAT_NN, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_D_REPEAT_NN, 1);
}
