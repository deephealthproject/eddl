/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/


#include "eddl/hardware/fpga/nn/fpga_nn.h"

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_repeat_nn       = 1;
char fpga_set_cpuemu_d_repeat_nn     = 1;


// -----------------------------------------------------------------
// repeat_nn
//
void fpga_cpuemu_repeat_nn(Tensor *A, Tensor *B, vector<int> size) {
    printf("fpga_cpuemu_repeat_nn not implemented yet\n");
    exit(1);
}

void fpga_repeat_nn(Tensor *A, Tensor *B, vector<int> size){
    _profile_fpga(_FPGA_REPEAT_NN, 0);
    if (fpga_set_cpuemu_repeat_nn == 1) {
        fpga_cpuemu_repeat_nn(A, B, size);
    } else {
        printf("fpga_repeat_nn not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_REPEAT_NN, 1);
}

// -----------------------------------------------------------------
// d_repeat_nn
//
void fpga_cpuemu_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size) {
    printf("fpga_cpuemu_d_repeat_nn not implemented yet\n");
    exit(1);
}

void fpga_d_repeat_nn(Tensor *D, Tensor *A, vector<int> size){
    _profile_fpga(_FPGA_D_REPEAT_NN, 0);
    if (fpga_set_cpuemu_d_repeat_nn == 1) {
        fpga_cpuemu_d_repeat_nn(D, A, size);
    } else {
        printf("fpga_equal not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_D_REPEAT_NN, 1);
}
