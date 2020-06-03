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

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_all           = 1;
char fpga_set_cpuemu_any           = 1;
char fpga_set_cpuemu_isfinite      = 1;
char fpga_set_cpuemu_isinf         = 1;
char fpga_set_cpuemu_isnan         = 1;
char fpga_set_cpuemu_isneginf      = 1;
char fpga_set_cpuemu_isposinf      = 1;
char fpga_set_cpuemu_logical_and   = 1;
char fpga_set_cpuemu_logical_or    = 1;
char fpga_set_cpuemu_logical_not   = 1;
char fpga_set_cpuemu_logical_xor   = 1;
char fpga_set_cpuemu_allclose      = 1;
char fpga_set_cpuemu_isclose       = 1;
char fpga_set_cpuemu_greater       = 1;
char fpga_set_cpuemu_greater_equal = 1;
char fpga_set_cpuemu_less          = 1;
char fpga_set_cpuemu_less_equal    = 1;
char fpga_set_cpuemu_equal         = 1;
char fpga_set_cpuemu_not_equal     = 1;
char fpga_set_cpuemu_equal2        = 1;

// FPGA: Logic functions: Truth value testing

// -----------------------------------------------------------------
// all
//
void fpga_cpuemu_all(Tensor *A) {
    printf("fpga_cpuemu_all not implemented yet\n");
    exit(1);
}

bool fpga_all(Tensor *A){
    _profile_fpga(_FPGA_ALL, 0);
    if (fpga_set_cpuemu_all == 1) {
        fpga_cpuemu_all(A);
    } else {
        printf("fpga_all not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ALL, 1);
    return 0;
}

// -----------------------------------------------------------------
// any
//
void fpga_cpuemu_any(Tensor *A) {
    printf("fpga_cpuemu_any not implemented yet\n");
    exit(1);
}

bool fpga_any(Tensor *A){
    bool res = false;
    _profile_fpga(_FPGA_ANY, 0);
    if (fpga_set_cpuemu_any == 1) {
        fpga_cpuemu_any(A);
    } else {
        printf("fpga_any not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ANY, 1);
    return res;
}

// CPU: Logic functions: Comparisons

// -----------------------------------------------------------------
// isfinite
//
void fpga_cpuemu_isfinite(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_isfinite not implemented yet\n");
    exit(1);
}

void fpga_isfinite(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISFINITE, 0);
    if (fpga_set_cpuemu_isfinite == 1) {
        fpga_cpuemu_isfinite(A, B);
    } else {
        printf("fpga_isfinite not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISFINITE, 1);
}

// -----------------------------------------------------------------
// isinf
//
void fpga_cpuemu_isinf(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_isinf not implemented yet\n");
    exit(1);
}

void fpga_isinf(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISINF, 0);
    if (fpga_set_cpuemu_isinf == 1) {
        fpga_cpuemu_isinf(A, B);
    } else {
        printf("fpga_isinf not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISINF, 1);
}

// -----------------------------------------------------------------
// isnan
//
void fpga_cpuemu_isnan(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_isnan not implemented yet\n");
    exit(1);
}

void fpga_isnan(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISNAN, 0);
    if (fpga_set_cpuemu_isnan == 1) {
        fpga_cpuemu_isnan(A, B);
    } else {
        printf("fpga_isnan not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISNAN, 1);
}

// -----------------------------------------------------------------
// isneginf
//
void fpga_cpuemu_isneginf(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_isneginf not implemented yet\n");
    exit(1);
}
void fpga_isneginf(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISNEGINF, 0);
    if (fpga_set_cpuemu_isneginf == 1) {
        fpga_cpuemu_isneginf(A, B);
    } else {
        printf("fpga_isneginf not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISNEGINF, 1);
}

// -----------------------------------------------------------------
// isposinf
//
void fpga_cpuemu_isposinf(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_isposinf not implemented yet\n");
    exit(1);
}

void fpga_isposinf(Tensor *A, Tensor* B){
    _profile_fpga(_FPGA_ISPOSINF, 0);
    if (fpga_set_cpuemu_isposinf == 1) {
        fpga_cpuemu_isposinf(A, B);
    } else {
        printf("fpga_isposinf not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISPOSINF, 1);
}

// CPU: Logic functions: Comparisons

// -----------------------------------------------------------------
// logical_and
//
void fpga_cpuemu_logical_and(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_logical_and not implemented yet\n");
    exit(1);
}

void fpga_logical_and(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LOGICAL_AND, 0);
    if (fpga_set_cpuemu_logical_and == 1) {
        fpga_cpuemu_logical_and(A, B, C);
    } else {
        printf("fpga_logical_and not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOGICAL_AND, 1);
}

// -----------------------------------------------------------------
// logical_or
//
void fpga_cpuemu_logical_or(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_logical_or not implemented yet\n");
    exit(1);
}

void fpga_logical_or(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LOGICAL_OR, 0);
    if (fpga_set_cpuemu_logical_or == 1) {
        fpga_cpuemu_logical_or(A, B, C);
    } else {
        printf("fpga_logical_or not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOGICAL_OR, 1);
}

// -----------------------------------------------------------------
// logical_not
//
void fpga_cpuemu_logical_not(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_logical_not not implemented yet\n");
    exit(1);
}

void fpga_logical_not(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_LOGICAL_NOT, 0);
    if (fpga_set_cpuemu_logical_not == 1) {
        fpga_cpuemu_logical_not(A, B);
    } else {
        printf("fpga_logical_not not implemented yet\n"); exit(1);
    }
    printf("fpga_logical_not not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGICAL_NOT, 1);
}

// -----------------------------------------------------------------
// logical_xor
//
void fpga_cpuemu_logical_xor(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_logical_xor not implemented yet\n");
    exit(1);
}

void fpga_logical_xor(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LOGICAL_XOR, 0);
    if (fpga_set_cpuemu_logical_xor == 1) {
        fpga_cpuemu_logical_xor(A, B, C);
    } else {
        printf("fpga_logical_xor not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOGICAL_XOR, 1);
}

// FPGA: Logic functions: Comparisons

// -----------------------------------------------------------------
// allclose
//
void fpga_cpuemu_allclose(Tensor *A, Tensor *B, float rotl, float atol, bool equal_nan) {
    printf("fpga_cpuemu_allclose not implemented yet\n");
    exit(1);
}

bool fpga_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    bool allclose = true;
    _profile_fpga(_FPGA_ALLCLOSE, 0);
    if (fpga_set_cpuemu_allclose == 1) {
        fpga_cpuemu_allclose(A, B, rtol, atol, equal_nan);
    } else {
        printf("fpga_allclose not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ALLCLOSE, 1);
    return allclose;
}

// -----------------------------------------------------------------
// isclose
//
void fpga_cpuemu_isclose(Tensor *A, Tensor *B, float rotl, float atol, bool equal_nan) {
    printf("fpga_cpuemu_isclose not implemented yet\n");
    exit(1);
}

void fpga_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    _profile_fpga(_FPGA_ISCLOSE, 0);
    if (fpga_set_cpuemu_isclose == 1) {
        fpga_cpuemu_isclose(A, B, rtol, atol, equal_nan);
    } else {
        printf("fpga_isclose not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISCLOSE, 1);
}

// -----------------------------------------------------------------
// greater
//
void fpga_cpuemu_greater(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_greater not implemented yet\n");
    exit(1);
}

void fpga_greater(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_GREATER, 0);
    if (fpga_set_cpuemu_greater == 1) {
        fpga_cpuemu_greater(A, B, C);
    } else {
        printf("fpga_greater not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_GREATER, 1);
}

// -----------------------------------------------------------------
// greater_equal
//
void fpga_cpuemu_greater_equal(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_greater_equal not implemented yet\n");
    exit(1);
}

void fpga_greater_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_GREATER_EQUAL, 0);
    if (fpga_set_cpuemu_greater_equal == 1) {
        fpga_cpuemu_greater_equal(A, B, C);
    } else {
        printf("fpga_greater_equal not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_GREATER_EQUAL, 1);
}

// -----------------------------------------------------------------
// less
//
void fpga_cpuemu_less(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_less not implemented yet\n");
    exit(1);
}

void fpga_less(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LESS, 0);
    if (fpga_set_cpuemu_less == 1) {
        fpga_cpuemu_less(A, B, C);
    } else {
        printf("fpga_less not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LESS, 1);
}

// -----------------------------------------------------------------
// less_equal
//
void fpga_cpuemu_less_equal(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_less_equal not implemented yet\n");
    exit(1);
}

void fpga_less_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_LESS_EQUAL, 0);
    if (fpga_set_cpuemu_less_equal == 1) {
        fpga_cpuemu_less_equal(A, B, C);
    } else {
        printf("fpga_less_equal not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LESS_EQUAL, 1);
}

// -----------------------------------------------------------------
// equal
//
void fpga_cpuemu_equal(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_equal not implemented yet\n");
    exit(1);
}

void fpga_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_EQUAL, 0);
    if (fpga_set_cpuemu_equal == 1) {
        fpga_cpuemu_equal(A, B, C);
    } else {
        printf("fpga_equal not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_EQUAL, 1);
}

// -----------------------------------------------------------------
// not_equal
//
void fpga_cpuemu_not_equal(Tensor *A, Tensor *B, Tensor *C) {
    printf("fpga_cpuemu_not_equal not implemented yet\n");
    exit(1);
}

void fpga_not_equal(Tensor *A, Tensor *B, Tensor *C){
    _profile_fpga(_FPGA_NOT_EQUAL, 0);
    if (fpga_set_cpuemu_not_equal == 1) {
        fpga_cpuemu_not_equal(A, B, C);
    } else {
        printf("fpga_not_equal not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_NOT_EQUAL, 1);
}

// -----------------------------------------------------------------
// equal2
//
void fpga_cpuemu_equal2(Tensor *A, Tensor *B, float epsilon) {
    printf("fpga_cpuemu_equal2 not implemented yet\n");
    exit(1);
}

int fpga_equal2(Tensor *A, Tensor *B, float epsilon){
  _profile_fpga(_FPGA_EQUAL2, 0);
    if (fpga_set_cpuemu_equal2 == 1) {
        fpga_cpuemu_equal2(A, B, epsilon);
    } else {
        printf("fpga_equal2 not implemented yet\n"); exit(1);
    }
  _profile_fpga(_FPGA_EQUAL2, 1);
  return 1;
}
