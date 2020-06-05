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

extern cl::CommandQueue q;
extern cl::Kernel mult2D;
extern cl::Kernel sum2D_rowwise;

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_abs_          = 1;
char fpga_set_cpuemu_acos_         = 1;
char fpga_set_cpuemu_add_          = 1;
char fpga_set_cpuemu_asin_         = 1;
char fpga_set_cpuemu_atan_         = 1;
char fpga_set_cpuemu_ceil_         = 1;
char fpga_set_cpuemu_clamp_        = 1;
char fpga_set_cpuemu_cos_          = 1;
char fpga_set_cpuemu_cosh_         = 1;
char fpga_set_cpuemu_exp_          = 1;
char fpga_set_cpuemu_floor_        = 1;
char fpga_set_cpuemu_inv_          = 1;
char fpga_set_cpuemu_log_          = 1;
char fpga_set_cpuemu_log2_         = 1;
char fpga_set_cpuemu_log10_        = 1;
char fpga_set_cpuemu_logn_         = 1;
char fpga_set_cpuemu_mod_          = 1;
char fpga_set_cpuemu_mult_         = 1;
char fpga_set_cpuemu_normalize_    = 1;
char fpga_set_cpuemu_pow_          = 1;
char fpga_set_cpuemu_powb_         = 1;
char fpga_set_cpuemu_reciprocal_   = 1;
char fpga_set_cpuemu_remainder_    = 1;
char fpga_set_cpuemu_round_        = 1;
char fpga_set_cpuemu_rsqrt_        = 1;
char fpga_set_cpuemu_sigmoid_      = 1;
char fpga_set_cpuemu_sign_         = 1;
char fpga_set_cpuemu_sin_          = 1;
char fpga_set_cpuemu_sinh_         = 1;
char fpga_set_cpuemu_sqr_          = 1;
char fpga_set_cpuemu_sqrt_         = 1;
char fpga_set_cpuemu_tan_          = 1;
char fpga_set_cpuemu_tanh_         = 1;
char fpga_set_cpuemu_trunc_        = 1;
char fpga_set_cpuemu_add           = 1;
char fpga_set_cpuemu_inc           = 1;
char fpga_set_cpuemu_mult2D        = 1;
char fpga_set_cpuemu_el_div        = 1;
char fpga_set_cpuemu_el_mult       = 1;
char fpga_set_cpuemu_sign2         = 1;
char fpga_set_cpuemu_sum2D_rowwise = 1;
char fpga_set_cpuemu_sum2D_colwise = 1;
char fpga_set_cpuemu_max           = 1;
char fpga_set_cpuemu_min           = 1;
char fpga_set_cpuemu_sum           = 1;
char fpga_set_cpuemu_sum_abs       = 1;

// CPU: Math (in-place) ********************************************

// -----------------------------------------------------------------
// abs_
//
void fpga_cpuemu_abs_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_abs_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_abs_(Tensor *A) {
    _profile_fpga(_FPGA_ABS_, 0);

    if (fpga_set_cpuemu_abs_ == 1) {
        fpga_cpuemu_abs_(A);
    } else {
        printf("fpga_abs_ not implemented yet\n"); exit(1);
    }

    _profile_fpga(_FPGA_ABS_, 1);
}

// -----------------------------------------------------------------
// acos_
//
void fpga_cpuemu_acos_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_acos_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_acos_(Tensor *A){
    _profile_fpga(_FPGA_ACOS_, 0);

    if (fpga_set_cpuemu_acos_ == 1) {
        fpga_cpuemu_acos_(A);
    } else {
        printf("fpga_acos_ not implemented yet\n"); exit(1);
    }

    _profile_fpga(_FPGA_ACOS_, 1);
}

// -----------------------------------------------------------------
// add_
//
void fpga_cpuemu_add_(Tensor *A, float v) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_add_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_add_(Tensor *A, float v) {
    _profile_fpga(_FPGA_ADD_, 0);
    if (fpga_set_cpuemu_add_ == 1) {
        fpga_cpuemu_add_(A, v);
    } else {
        printf("fpga_add_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ADD_, 1);
}

// -----------------------------------------------------------------
// asin_
//
void fpga_cpuemu_asin_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_asin_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_asin_(Tensor *A){
    _profile_fpga(_FPGA_ASIN_, 0);
    if (fpga_set_cpuemu_asin_ == 1) {
        fpga_cpuemu_asin_(A);
    } else {
        printf("fpga_asin_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ASIN_, 1);
}

// -----------------------------------------------------------------
// atan_
//
void fpga_cpuemu_atan_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_atan_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_atan_(Tensor *A){
    _profile_fpga(_FPGA_ATAN_, 0);
    if (fpga_set_cpuemu_atan_ == 1) {
        fpga_cpuemu_atan_(A);
    } else {
        printf("fpga_atan_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ATAN_, 1);
}

// -----------------------------------------------------------------
// ceil_
//
void fpga_cpuemu_ceil_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_ceil_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_ceil_(Tensor *A){
    _profile_fpga(_FPGA_CEIL_, 0);
    if (fpga_set_cpuemu_ceil_ == 1) {
        fpga_cpuemu_ceil_(A);
    } else {
        printf("fpga_ceil_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CEIL_, 1);
}

// -----------------------------------------------------------------
// clamp_
//
void fpga_cpuemu_clamp_(Tensor *A, float min, float max) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_clamp_(A, min, max);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_clamp_(Tensor *A, float min, float max){
    _profile_fpga(_FPGA_CLAMP_, 0);
    if (fpga_set_cpuemu_clamp_ == 1) {
        fpga_cpuemu_clamp_(A, min, max);
    } else {
        printf("fpga_clamp_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CLAMP_, 1);
}

// -----------------------------------------------------------------
// cos_
//
void fpga_cpuemu_cos_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_cos_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_cos_(Tensor *A){
    _profile_fpga(_FPGA_COS_, 0);
    if (fpga_set_cpuemu_cos_ == 1) {
        fpga_cpuemu_cos_(A);
    } else {
        printf("fpga_cos_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_COS_, 1);
}

// -----------------------------------------------------------------
// cosh_
//
void fpga_cpuemu_cosh_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_cosh_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_cosh_(Tensor *A){
    _profile_fpga(_FPGA_COSH_, 0);
    if (fpga_set_cpuemu_cosh_ == 1) {
        fpga_cpuemu_cosh_(A);
    } else {
        printf("fpga_cosh_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_COSH_, 1);
}

// -----------------------------------------------------------------
// exp_
//
void fpga_cpuemu_exp_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_exp_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_exp_(Tensor *A) {
    _profile_fpga(_FPGA_EXP_, 0);
    if (fpga_set_cpuemu_exp_ == 1) {
        fpga_cpuemu_exp_(A);
    } else {
        printf("fpga_exp_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_EXP_, 1);
}

// -----------------------------------------------------------------
// floor_
//
void fpga_cpuemu_floor_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_floor_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_floor_(Tensor *A){
    _profile_fpga(_FPGA_FLOOR_, 0);
    if (fpga_set_cpuemu_floor_ == 1) {
        fpga_cpuemu_floor_(A);
    } else {
        printf("fpga_floor_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_FLOOR_, 1);
}

// -----------------------------------------------------------------
// inv_
//
void fpga_cpuemu_inv_(Tensor *A, float v) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_inv_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_inv_(Tensor *A, float v){
    _profile_fpga(_FPGA_INV_, 0);
    if (fpga_set_cpuemu_inv_ == 1) {
        fpga_cpuemu_inv_(A, v);
    } else {
        printf("fpga_inv_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_INV_, 1);
}

// -----------------------------------------------------------------
// log_
//
void fpga_cpuemu_log_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_log_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_log_(Tensor *A) {
    _profile_fpga(_FPGA_LOG_, 0);
    if (fpga_set_cpuemu_log_ == 1) {
        fpga_cpuemu_log_(A);
    } else {
        printf("fpga_log_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOG_, 1);
}

// -----------------------------------------------------------------
// log2_
//
void fpga_cpuemu_log2_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_log2_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_log2_(Tensor *A) {
    _profile_fpga(_FPGA_LOG2_, 0);
    if (fpga_set_cpuemu_log2_ == 1) {
        fpga_cpuemu_log2_(A);
    } else {
        printf("fpga_log2_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOG2_, 1);
}

// -----------------------------------------------------------------
// log10_
//
void fpga_cpuemu_log10_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_log10_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_log10_(Tensor *A) {
    _profile_fpga(_FPGA_LOG10_, 0);
    if (fpga_set_cpuemu_log10_ == 1) {
        fpga_cpuemu_log10_(A);
    } else {
        printf("fpga_log10_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOG10_, 1);
}

// -----------------------------------------------------------------
// logn_
//
void fpga_cpuemu_logn_(Tensor *A, float n) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_logn_(A, n);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_logn_(Tensor *A, float n) {
    _profile_fpga(_FPGA_LOGN_, 0);
    if (fpga_set_cpuemu_logn_ == 1) {
        fpga_cpuemu_logn_(A, n);
    } else {
        printf("fpga_logn_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_LOGN_, 1);
}

// -----------------------------------------------------------------
// mod_
//
void fpga_cpuemu_mod_(Tensor *A, float v) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_mod_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_mod_(Tensor *A, float v){
    _profile_fpga(_FPGA_MOD_, 0);
    if (fpga_set_cpuemu_mod_ == 1) {
        fpga_cpuemu_mod_(A, v);
    } else {
        printf("fpga_mod_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_MOD_, 1);
}

// -----------------------------------------------------------------
// mult_
//
void fpga_cpuemu_mult_(Tensor *A, float v) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_mult_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_mult_(Tensor *A, float v) {
    _profile_fpga(_FPGA_MULT_, 0);
    if (fpga_set_cpuemu_mult_ == 1) {
        fpga_cpuemu_mult_(A, v);
    } else {
        printf("fpga_mult_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_MULT_, 1);
}

// -----------------------------------------------------------------
// normalize_
//
void fpga_cpuemu_normalize_(Tensor *A, float min, float max) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_normalize_(A, min, max);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_normalize_(Tensor *A, float min, float max){
    _profile_fpga(_FPGA_NORMALIZE_, 0);
    if (fpga_set_cpuemu_normalize_ == 1) {
        fpga_cpuemu_normalize_(A, min, max);
    } else {
        printf("fpga_normalize_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_NORMALIZE_, 1);
}

// -----------------------------------------------------------------
// pow_
//
void fpga_cpuemu_pow_(Tensor *A, float exp) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_pow_(A, exp);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_pow_(Tensor *A, float exp) {
    _profile_fpga(_FPGA_POW_, 0);
    if (fpga_set_cpuemu_pow_ == 1) {
        fpga_cpuemu_pow_(A, exp);
    } else {
        printf("fpga_pow_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_POW_, 1);
}

// -----------------------------------------------------------------
// powb_
//
void fpga_cpuemu_powb_(Tensor *A, float base) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_powb_(A, base);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_powb_(Tensor *A, float base) {
    _profile_fpga(_FPGA_POWB_, 0);
    if (fpga_set_cpuemu_powb_ == 1) {
        fpga_cpuemu_powb_(A, base);
    } else {
        printf("fpga_powb_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_POWB_, 1);
}

// -----------------------------------------------------------------
// reciprocal_
//
void fpga_cpuemu_reciprocal_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_reciprocal_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_reciprocal_(Tensor *A) {
    _profile_fpga(_FPGA_RECIPROCAL_, 0);
    if (fpga_set_cpuemu_reciprocal_ == 1) {
        fpga_cpuemu_reciprocal_(A);
    } else {
        printf("fpga_reciprocal_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_RECIPROCAL_, 1);
}

// -----------------------------------------------------------------
// remainder_
//
void fpga_cpuemu_remainder_(Tensor *A, float v) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_remainder_(A, v);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_remainder_(Tensor *A, float v) {
    _profile_fpga(_FPGA_REMAINDER_, 0);
    if (fpga_set_cpuemu_remainder_ == 1) {
        fpga_cpuemu_remainder_(A, v);
    } else {
        printf("fpga_remainder_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_REMAINDER_, 1);
}

// -----------------------------------------------------------------
// round_
//
void fpga_cpuemu_round_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_round_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_round_(Tensor *A){
    _profile_fpga(_FPGA_ROUND_, 0);
    if (fpga_set_cpuemu_round_ == 1) {
        fpga_cpuemu_round_(A);
    } else {
        printf("fpga_round_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ROUND_, 1);
}

// -----------------------------------------------------------------
// rsqrt_
//
void fpga_cpuemu_rsqrt_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_rsqrt_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_rsqrt_(Tensor *A){
    _profile_fpga(_FPGA_RSQRT_, 0);
    if (fpga_set_cpuemu_rsqrt_ == 1) {
        fpga_cpuemu_rsqrt_(A);
    } else {
        printf("fpga_rsqrt_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_RSQRT_, 1);
}

// -----------------------------------------------------------------
// sigmoid_
//
void fpga_cpuemu_sigmoid_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sigmoid_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_sigmoid_(Tensor *A){
    _profile_fpga(_FPGA_SIGMOID_, 0);
    if (fpga_set_cpuemu_sigmoid_ == 1) {
        fpga_cpuemu_sigmoid_(A);
    } else {
        printf("fpga_sigmoid_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SIGMOID_, 1);
}

// -----------------------------------------------------------------
// sign_
//
void fpga_cpuemu_sign_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sign_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_sign_(Tensor *A){
    _profile_fpga(_FPGA_SIGN_, 0);
    if (fpga_set_cpuemu_sign_ == 1) {
        fpga_cpuemu_sign_(A);
    } else {
        printf("fpga_sign_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SIGN_, 1);
}

// -----------------------------------------------------------------
// sin_
//
void fpga_cpuemu_sin_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sin_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_sin_(Tensor *A){
    _profile_fpga(_FPGA_SIN_, 0);
    if (fpga_set_cpuemu_sin_ == 1) {
        fpga_cpuemu_sin_(A);
    } else {
        printf("fpga_sin_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SIN_, 1);
}

// -----------------------------------------------------------------
// sinh_
//
void fpga_cpuemu_sinh_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sinh_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_sinh_(Tensor *A){
    _profile_fpga(_FPGA_SINH_, 0);
    if (fpga_set_cpuemu_sinh_ == 1) {
        fpga_cpuemu_sinh_(A);
    } else {
        printf("fpga_sinh_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINH_, 1);
}

// -----------------------------------------------------------------
// sqr_
//
void fpga_cpuemu_sqr_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sqr_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_sqr_(Tensor *A) {
    _profile_fpga(_FPGA_SQR_, 0);
    if (fpga_set_cpuemu_sqr_ == 1) {
        fpga_cpuemu_sqr_(A);
    } else {
        printf("fpga_sqr_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SQR_, 1);
}

// -----------------------------------------------------------------
// sqrt_
//
void fpga_cpuemu_sqrt_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sqrt_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_sqrt_(Tensor *A) {
    _profile_fpga(_FPGA_SQRT_, 0);
    if (fpga_set_cpuemu_sqrt_ == 1) {
        fpga_cpuemu_sqrt_(A);
    } else {
        printf("fpga_sqrt_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SQRT_, 1);
}

// -----------------------------------------------------------------
// tan_
//
void fpga_cpuemu_tan_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_tan_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_tan_(Tensor *A){
    _profile_fpga(_FPGA_TAN_, 0);
    if (fpga_set_cpuemu_tan_ == 1) {
        fpga_cpuemu_tan_(A);
    } else {
        printf("fpga_tanh_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_TAN_, 1);
}

// -----------------------------------------------------------------
// tanh_
//
void fpga_cpuemu_tanh_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_tanh_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_tanh_(Tensor *A){
    _profile_fpga(_FPGA_TANH_, 0);
    if (fpga_set_cpuemu_tanh_ == 1) {
        fpga_cpuemu_tanh_(A);
    } else {
        printf("fpga_tanh_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_TANH_, 1);
}

// -----------------------------------------------------------------
// trunc_
//
void fpga_cpuemu_trunc_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_trunc_(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_trunc_(Tensor *A){
    _profile_fpga(_FPGA_TRUNC_, 0);
    if (fpga_set_cpuemu_trunc_ == 1) {
        fpga_cpuemu_trunc_(A);
    } else {
        printf("fpga_trunc_ not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_TRUNC_, 1);
}

// FPGA: Math (static) ***************************

// -----------------------------------------------------------------
// add
//
void fpga_cpuemu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_add(scA, A, scB, B, C, incC);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
    _profile_fpga(_FPGA_ADD, 0);
    if (fpga_set_cpuemu_add == 1) {
        fpga_cpuemu_add(scA, A, scB, B, C, incC);
    } else {
        printf("fpga_add not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ADD, 1);
}

// -----------------------------------------------------------------
// inc
//
void fpga_cpuemu_inc(Tensor *A, Tensor *B) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_inc(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_inc(Tensor *A, Tensor *B) {
  _profile_fpga(_FPGA_INC, 0);
  B->tsem->lock();               // why locks?
  if (fpga_set_cpuemu_inc == 1) {
    fpga_cpuemu_inc(A, B);
  } else {
    printf("fpga_inc not implemented yet\n"); exit(1);
  }
  B->tsem->unlock();
  _profile_fpga(_FPGA_INC, 1);
}

// -----------------------------------------------------------------
// mult2D
//
void fpga_cpuemu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_mult2D(A, tA, B, tB, C, incC);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    _profile_fpga(_FPGA_MULT2D, 0);
    if (fpga_set_cpuemu_mult2D == 1) {
      fpga_cpuemu_mult2D(A, tA, B, tB, C, incC);
    } else {
      printf("fpga_mult2D not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_MULT2D, 1);
}

// -----------------------------------------------------------------
// el_div
//
void fpga_cpuemu_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_el_div(A, B, C, incC);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile_fpga(_FPGA_EL_DIV, 0);
  if (fpga_set_cpuemu_el_div == 1) {
    fpga_cpuemu_el_div(A, B, C, incC);
  } else {
    printf("fpga_el_div not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_EL_DIV, 1);
}

// -----------------------------------------------------------------
// el_mult
//
void fpga_cpuemu_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_el_mult(A, B, C, incC);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile_fpga(_FPGA_EL_MULT, 0);
  if (fpga_set_cpuemu_el_mult == 1) {
    fpga_cpuemu_el_mult(A, B, C, incC);
  } else {
    printf("fpga_el_mult not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_EL_MULT, 1);
}

// -----------------------------------------------------------------
// sign2
//
void fpga_cpuemu_sign2(Tensor *A, Tensor *B) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sign2(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sign2(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SIGN2, 0);
  if (fpga_set_cpuemu_sign2 == 1) {
    fpga_cpuemu_sign2(A, B);
  } else {
    printf("fpga_sign2 not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_SIGN2, 1);
   
}

// -----------------------------------------------------------------
// sum2D_rowwise
//
void fpga_cpuemu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_sum2D_rowwise(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile_fpga(_FPGA_SUM2D_ROWWISE, 0);
  if (fpga_set_cpuemu_sum2D_rowwise == 1) {
    fpga_cpuemu_sum2D_rowwise(A, B, C);
  } else {
    cl_int err;
    cl::Event event;
    OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(2, (C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(3, A->shape[0]));
    OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(4, A->shape[1]));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sum2D_rowwise, NULL, &event));
    q.finish();
  }
  _profile_fpga(_FPGA_SUM2D_ROWWISE, 1);
}

// -----------------------------------------------------------------
// sum2D_colwise
//
void fpga_cpuemu_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  int Csize = C->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  if (C->ptr == NULL) C->ptr = (float *)malloc(Csize);
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_sum2D_colwise(A, B, C);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile_fpga(_FPGA_SUM2D_COLWISE, 0);
  if (fpga_set_cpuemu_sum2D_colwise == 1) {
    fpga_cpuemu_sum2D_colwise(A, B, C);
  } else {
    printf("fpga_sum2D_colwise not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_SUM2D_COLWISE, 1);
}

// FPGA: Should be reductions ***************************

// -----------------------------------------------------------------
// fpga_max
//
float fpga_cpuemu_max(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  float ret = cpu_max(A);
  return ret;
}

float fpga_max(Tensor *A){
  float ret;
  _profile_fpga(_FPGA_MAX, 0);
  if (fpga_set_cpuemu_max == 1) {
    ret = fpga_cpuemu_max(A);
  } else {
    printf("fpga_max not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_MAX, 1);
  return ret;
}

// -----------------------------------------------------------------
// fpga_min
//
float fpga_cpuemu_min(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  float ret = cpu_min(A);
  return ret;
}

float fpga_min(Tensor *A){
  float ret;
  _profile_fpga(_FPGA_MIN, 0);
  if (fpga_set_cpuemu_min == 1) {
    ret = fpga_cpuemu_min(A);
  } else {
    printf("fpga_min not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_MIN, 1);
  return ret; 
}

// -----------------------------------------------------------------
// fpga_sum
//
float fpga_cpuemu_sum(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  float ret = cpu_sum(A);
  return ret;
}

float fpga_sum(Tensor *A) {
  float ret;
  _profile_fpga(_FPGA_SUM, 0);
  if (fpga_set_cpuemu_sum == 1) {
    ret = fpga_cpuemu_sum(A);
  } else {
    printf("fpga_sum not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_SUM, 1);
  return ret;
}

// -----------------------------------------------------------------
// sum_abs
//
float fpga_cpuemu_sum_abs(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  float ret = cpu_sum_abs(A);
  return ret;
}

float fpga_sum_abs(Tensor *A) {
  float ret;
  _profile_fpga(_FPGA_SUM_ABS, 0);
  if (fpga_set_cpuemu_sum_abs == 1) {
    ret = fpga_cpuemu_sum_abs(A);
  } else {
    printf("fpga_abs not implemented yet\n"); exit(1);
  }
  _profile_fpga(_FPGA_SUM_ABS, 1);
  return ret;
}
