/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/hardware/fpga/fpga_hw.h"
#include <limits>

// FPGA: Logic functions: Truth value testing
bool fpga_all(Tensor *A){
    _profile_fpga(_FPGA_ALL, 0);
    printf("fpga_all not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ALL, 1);
    return 0;
}

bool fpga_any(Tensor *A){
    bool res = false;
    _profile_fpga(_FPGA_ANY, 0);
    printf("fpga_any not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ANY, 1);
    return res;
}

// CPU: Logic functions: Comparisons
void fpga_isfinite(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISFINITE, 0);
    printf("fpga_isfinite not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ISFINITE, 1);
}

void fpga_isinf(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISINF, 0);
    printf("fpga_isinf not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ISINF, 1);
}

void fpga_isnan(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISNAN, 0);
    printf("fpga_isnan not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ISNAN, 1);
}

void fpga_isneginf(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISNEGINF, 0);
    printf("fpga_isneginf not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ISNEGINF, 1);
}

void fpga_isposinf(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISPOSINF, 0);
    printf("fpga_isposinf not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ISPOSINF, 1);
}


// CPU: Logic functions: Comparisons
void fpga_logical_and(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LOGICAL_AND, 0);
    printf("fpga_logical_and not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGICAL_AND, 1);
}

void fpga_logical_or(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LOGICAL_OR, 0);
    printf("fpga_logical_or not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGICAL_OR, 1);
}

void fpga_logical_not(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_LOGICAL_NOT, 0);
    printf("fpga_logical_not not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGICAL_NOT, 1);
}

void fpga_logical_xor(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LOGICAL_XOR, 0);
    printf("fpga_logical_xor not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGICAL_XOR, 1);
}

// FPGA: Logic functions: Comparisons
bool fpga_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    bool allclose = true;
    _profile_fpga(_FPGA_ALLCLOSE, 0);
    printf("fpga_allclose not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ALLCLOSE, 1);
    return allclose;
}

void fpga_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    _profile_fpga(_FPGA_ISCLOSE, 0);
    printf("fpga_isclose not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ISCLOSE, 1);
}

void fpga_greater(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_GREATER, 0);
    printf("fpga_greate not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_GREATER, 1);
}

void fpga_greater_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_GREATER_EQUAL, 0);
    printf("fpga_greater_equal not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_GREATER_EQUAL, 1);
}

void fpga_less(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LESS, 0);
    printf("fpga_less not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LESS, 1);
}

void fpga_less_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LESS_EQUAL, 0);
    printf("fpga_less_equal not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LESS_EQUAL, 1);
}

void fpga_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_EQUAL, 0);
    printf("fpga_equal not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_EQUAL, 1);
}

void fpga_not_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_NOT_EQUAL, 0);
    printf("fpga_not_equal not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_NOT_EQUAL, 1);
}

int fpga_equal2(Tensor *A, Tensor *B, float epsilon){
  _profile_fpga(_FPGA_EQUAL2, 0);
  printf("fpga_equal2 not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_EQUAL2, 1);
  return 1;
}
