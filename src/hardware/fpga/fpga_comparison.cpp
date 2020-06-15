/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_hw.h"
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
bool fpga_cpuemu_all(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  bool ret = cpu_all(A);
  return ret;
}

bool fpga_all(Tensor *A){
    int ret;
    _profile_fpga(_FPGA_ALL, 0);
    if (fpga_set_cpuemu_all == 1) {
        ret = fpga_cpuemu_all(A);
    } else {
        printf("fpga_all not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ALL, 1);
    return ret;
}

// -----------------------------------------------------------------
// any
//
bool fpga_cpuemu_any(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  bool ret = cpu_any(A);
  return ret;
}

bool fpga_any(Tensor *A){
    bool res = false;
    _profile_fpga(_FPGA_ANY, 0);
    if (fpga_set_cpuemu_any == 1) {
        res = fpga_cpuemu_any(A);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_isfinite(A, B);
  fpga_copy_to_fpga(B->ptr, B);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_isinf(A, B);
  fpga_copy_to_fpga(B->ptr, B);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_isnan(A, B);
  fpga_copy_to_fpga(B->ptr, B);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_isneginf(A, B);
  fpga_copy_to_fpga(B->ptr, B);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_isposinf(A, B);
  fpga_copy_to_fpga(B->ptr, B);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_logical_and(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_logical_or(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_logical_not(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_logical_not(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_LOGICAL_NOT, 0);
    if (fpga_set_cpuemu_logical_not == 1) {
        fpga_cpuemu_logical_not(A, B);
    } else {
        printf("fpga_logical_not not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOGICAL_NOT, 1);
}

// -----------------------------------------------------------------
// logical_xor
//
void fpga_cpuemu_logical_xor(Tensor *A, Tensor *B, Tensor *C) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_logical_xor(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
bool fpga_cpuemu_allclose(Tensor *A, Tensor *B, float rotl, float atol, bool equal_nan) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  bool ret = cpu_allclose(A, B, rotl, atol, equal_nan);
  return ret;
}

bool fpga_allclose(Tensor *A, Tensor *B, float rtol, float atol, bool equal_nan){
    bool allclose = true;
    _profile_fpga(_FPGA_ALLCLOSE, 0);
    if (fpga_set_cpuemu_allclose == 1) {
        allclose = fpga_cpuemu_allclose(A, B, rtol, atol, equal_nan);
    } else {
        printf("fpga_allclose not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ALLCLOSE, 1);
    return allclose;
}

// -----------------------------------------------------------------
// isclose
//
void fpga_cpuemu_isclose(Tensor *A, Tensor *B, Tensor *C, float rotl, float atol, bool equal_nan) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_isclose(A, B, C, rotl, atol, equal_nan);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_isclose(Tensor *A, Tensor *B, Tensor *C, float rtol, float atol, bool equal_nan){
    _profile_fpga(_FPGA_ISCLOSE, 0);
    if (fpga_set_cpuemu_isclose == 1) {
        fpga_cpuemu_isclose(A, B, C, rtol, atol, equal_nan);
    } else {
        printf("fpga_isclose not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ISCLOSE, 1);
}

// -----------------------------------------------------------------
// greater
//
void fpga_cpuemu_greater(Tensor *A, Tensor *B, Tensor *C) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_greater(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_greater_equal(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_less(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_less_equal(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_equal(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_not_equal(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
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
int fpga_cpuemu_equal2(Tensor *A, Tensor *B, float epsilon) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  int ret = cpu_equal2(A, B, epsilon);
  return ret;
}

int fpga_equal2(Tensor *A, Tensor *B, float epsilon){
  int ret;
  _profile_fpga(_FPGA_EQUAL2, 0);
    if (fpga_set_cpuemu_equal2 == 1) {
        ret = fpga_cpuemu_equal2(A, B, epsilon);
    } else {
        printf("fpga_equal2 not implemented yet\n"); exit(1);
    }
  _profile_fpga(_FPGA_EQUAL2, 1);
  return ret;
}
