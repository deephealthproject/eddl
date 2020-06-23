/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

extern cl::Context context;
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
char fpga_set_cpuemu_normalize_    = 1;
char fpga_set_cpuemu_pow_          = 1;
char fpga_set_cpuemu_powb_         = 1;
char fpga_set_cpuemu_reciprocal_   = 1;
char fpga_set_cpuemu_remainder_    = 1;
char fpga_set_cpuemu_round_        = 1;
char fpga_set_cpuemu_rsqrt_        = 1;
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
void fpga_cpuemu_abs(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_abs(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_abs(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_ABS_, 0);
#ifndef K_ENABLED_ABS_
    fpga_cpuemu_abs(A, B);
#else
    printf("Añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_abs_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_abs_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_abs_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_ABS_, 1);
}

// -----------------------------------------------------------------
// acos_
//
void fpga_cpuemu_acos(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_acos(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_acos(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_ACOS_, 0);
#ifndef K_ENABLED_ACOS_
    fpga_cpuemu_acos(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_acos_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_acos_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_acos_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_ACOS_, 1);
}

// -----------------------------------------------------------------
// add
//
void fpga_cpuemu_add(Tensor *A, Tensor *B, float v) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_add(A, B, v);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_add(Tensor *A, Tensor *B, float v) {
    _profile_fpga(_FPGA_ADD_, 0);
#ifndef K_ENABLED_ADD_
    fpga_cpuemu_add(A, B, v);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;


    OCL_CHECK(err, err = kernel_add_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add_.setArg(1, v));
    OCL_CHECK(err, err = kernel_add_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_add_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_ADD_, 1);
}

// -----------------------------------------------------------------
// asin
//
void fpga_cpuemu_asin(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_asin(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_asin(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_ASIN_, 0);
#ifndef K_ENABLED_ASIN_
    fpga_cpuemu_asin(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_asin_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_asin_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_asin_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_ASIN_, 1);
}

// -----------------------------------------------------------------
// atan
//
void fpga_cpuemu_atan(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_atan(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_atan(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_ATAN_, 0);
#ifndef K_ENABLED_ATAN_
    fpga_cpuemu_atan(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_atan_.setArg(0, *(A->fpga_ptr)))
    OCL_CHECK(err, err = kernel_atan_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_atan_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_ATAN_, 1);
}

// -----------------------------------------------------------------
// ceil
//
void fpga_cpuemu_ceil(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_ceil(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_ceil(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_CEIL_, 0);
#ifndef K_ENABLED_CEIL_
    fpga_cpuemu_ceil(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_ceil_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_ceil_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_ceil_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_CEIL_, 1);
}

// -----------------------------------------------------------------
// clamp
//
void fpga_cpuemu_clamp(Tensor *A, Tensor *B, float min, float max) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_clamp(A, B, min, max);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_clamp(Tensor *A, Tensor *B, float min, float max){
    _profile_fpga(_FPGA_CLAMP_, 0);
#ifndef K_ENABLED_CLAMP_
    fpga_cpuemu_clamp(A, B, min, max);
#else
    printf("añadir tensorB\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_clamp_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_clamp_.setArg(1, min));
    OCL_CHECK(err, err = kernel_clamp_.setArg(2, max));
    OCL_CHECK(err, err = kernel_clamp_.setArg(3, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_clamp_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_CLAMP_, 1);
}

// -----------------------------------------------------------------
// cos
//
void fpga_cpuemu_cos(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_cos(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_cos(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_COS_, 0);
#ifndef K_ENABLED_COS_
    fpga_cpuemu_cos(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_cos_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cos_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_cos_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_COS_, 1);
}

// -----------------------------------------------------------------
// cosh
//
void fpga_cpuemu_cosh(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_cosh(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_cosh(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_COSH_, 0);
#ifndef K_ENABLED_COSH_
    fpga_cpuemu_cosh(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_cosh_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_cosh_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_cosh_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_COSH_, 1);
}

// -----------------------------------------------------------------
// exp
//
void fpga_cpuemu_exp(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_exp(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_exp(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_EXP_, 0);
#ifndef K_ENABLED_EXP_
    fpga_cpuemu_exp(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_exp_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_exp_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_exp_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_EXP_, 1);
}

// -----------------------------------------------------------------
// floor
//
void fpga_cpuemu_floor(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_floor(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_floor(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_FLOOR_, 0);
#ifndef K_ENABLED_FLOOR_
    fpga_cpuemu_floor(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_floor_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_floor_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_floor_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_FLOOR_, 1);
}

// -----------------------------------------------------------------
// inv
//
void fpga_cpuemu_inv(Tensor *A, Tensor *B, float v) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_inv(A, B, v);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_inv(Tensor *A, Tensor *B, float v){
    _profile_fpga(_FPGA_INV_, 0);
#ifndef K_ENABLED_INV_
    fpga_cpuemu_inv(A, B, v);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_inv_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_inv_.setArg(1, v));
    OCL_CHECK(err, err = kernel_inv_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_inv_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_INV_, 1);
}

// -----------------------------------------------------------------
// log
//
void fpga_cpuemu_log(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_log(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_log(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_LOG_, 0);
#ifndef K_ENABLED_LOG_
    fpga_cpuemu_log(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_log_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_log_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_log_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_LOG_, 1);
}

// -----------------------------------------------------------------
// log2
//
void fpga_cpuemu_log2(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_log2(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_log2(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_LOG2_, 0);
#ifndef K_ENABLED_log2_
    fpga_cpuemu_log2(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_log2_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_log2_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_log2_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_LOG2_, 1);
}

// -----------------------------------------------------------------
// log10
//
void fpga_cpuemu_log10(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_log10(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_log10(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_LOG10_, 0);
#ifndef K_ENABLED_LOG10_
    fpga_cpuemu_log10(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_log10_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_log10_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_log10_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_LOG10_, 1);
}

// -----------------------------------------------------------------
// logn
//
void fpga_cpuemu_logn(Tensor *A, Tensor *B, float n) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_logn(A, B, n);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_logn(Tensor *A, Tensor *B, float n) {
    _profile_fpga(_FPGA_LOGN_, 0);
#ifndef K_ENABLED_LOGN_
    fpga_cpuemu_logn(A, B, n);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_logn_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_logn_.setArg(1, n));
    OCL_CHECK(err, err = kernel_logn_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_logn_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_LOGN_, 1);
}

// -----------------------------------------------------------------
// mod
//
void fpga_cpuemu_mod(Tensor *A, Tensor *B, float v) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_mod(A, B, v);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_mod(Tensor *A, Tensor *B, float v){
    _profile_fpga(_FPGA_MOD_, 0);
#ifndef K_ENABLED_MOD_
    fpga_cpuemu_mod(A, B, v);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_mod_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_mod_.setArg(1, v));
    OCL_CHECK(err, err = kernel_mod_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_mod_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_MOD_, 1);
}

// -----------------------------------------------------------------
// mult
//
void fpga_cpuemu_mult(Tensor *A, Tensor *B, float v) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_mult(A, B, v);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_mult(Tensor *A, Tensor *B, float v) {
    _profile_fpga(_FPGA_MULT_, 0);
    _profile_fpga_tensor(A);
#ifndef K_ENABLED_MULT_
    fpga_cpuemu_mult(A, B, v);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_mult_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_mult_.setArg(1, v));
    OCL_CHECK(err, err = kernel_mult_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_mult_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_MULT_, 1);
}

// -----------------------------------------------------------------
// normalize
//
void fpga_cpuemu_normalize(Tensor *A, Tensor *B, float min, float max) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_normalize(A, B, min, max);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_normalize(Tensor *A, Tensor *B, float min, float max){
    _profile_fpga(_FPGA_NORMALIZE_, 0);
#ifndef K_ENABLED_NORMALIZE_
      fpga_cpuemu_normalize(A, B, min, max);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_normalize_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_normalize_.setArg(1, min));
    OCL_CHECK(err, err = kernel_normalize_.setArg(2, max));
    OCL_CHECK(err, err = kernel_normalize_.setArg(3, (long int)A->size));
    OCL_CHECK(err, err = kernel_normalize_.setArg(4, A->min()));
    OCL_CHECK(err, err = kernel_normalize_.setArg(5, A->max()));

    OCL_CHECK(err, err = q.enqueueTask(kernel_normalize_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_NORMALIZE_, 1);
}

// -----------------------------------------------------------------
// pow
//
void fpga_cpuemu_pow(Tensor *A, Tensor *B, float exp) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_pow(A, B, exp);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_pow(Tensor *A, Tensor *B, float exp) {
    _profile_fpga(_FPGA_POW_, 0);
#ifndef K_ENABLED_POW_
    fpga_cpuemu_pow(A, B, exp);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_pow_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_pow_.setArg(1, exp));
    OCL_CHECK(err, err = kernel_pow_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_pow_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_POW_, 1);
}

// -----------------------------------------------------------------
// powb
//
void fpga_cpuemu_powb(Tensor *A, Tensor *B, float base) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_powb(A, B, base);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_powb(Tensor *A, Tensor *B, float base) {
    _profile_fpga(_FPGA_POWB_, 0);
#ifndef K_ENABLED_POWB_
    fpga_cpuemu_powb(A, B, base);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_powb_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_powb_.setArg(1, base));
    OCL_CHECK(err, err = kernel_powb_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_powb_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_POWB_, 1);
}

// -----------------------------------------------------------------
// reciprocal_
//
/*void fpga_cpuemu_reciprocal_(Tensor *A) {
  int Asize = A->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_reciprocal(A);
  fpga_copy_to_fpga(A->ptr, A);
}

void fpga_reciprocal_(Tensor *A) {
    _profile_fpga(_FPGA_RECIPROCAL_, 0);
#ifndef K_ENABLED_RECIPROCAL_
    fpga_cpuemu_reciprocal_(A);
#else
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_reciprocal_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_reciprocal_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_reciprocal_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_RECIPROCAL_, 1);
}*/

// -----------------------------------------------------------------
// remainder
//
void fpga_cpuemu_remainder(Tensor *A, Tensor *B, float v) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_remainder(A, B, v);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_remainder(Tensor *A, Tensor *B, float v) {
    _profile_fpga(_FPGA_REMAINDER_, 0);
#ifndef K_ENABLED_REMAINDER_
    fpga_cpuemu_remainder(A, B, v);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_remainder_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_remainder_.setArg(1, v));
    OCL_CHECK(err, err = kernel_remainder_.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_remainder_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_REMAINDER_, 1);
}

// -----------------------------------------------------------------
// round
//
void fpga_cpuemu_round(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_round(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_round(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_ROUND_, 0);
#ifndef K_ENABLED_ROUND_
    fpga_cpuemu_round(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_round_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_round_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_round_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_ROUND_, 1);
}

// -----------------------------------------------------------------
// rsqrt
//
void fpga_cpuemu_rsqrt(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_rsqrt(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_rsqrt(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_RSQRT_, 0);
#ifndef K_ENABLED_RSQRT_
    fpga_cpuemu_rsqrt(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_rsqrt_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_rsqrt_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_rsqrt_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_RSQRT_, 1);
}

// -----------------------------------------------------------------
// sin
//
void fpga_cpuemu_sin(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sin(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sin(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_SIN_, 0);
#ifndef K_ENABLED_SIN_
    fpga_cpuemu_sin(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_sin_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sin_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sin_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_SIN_, 1);
}

// -----------------------------------------------------------------
// sinh
//
void fpga_cpuemu_sinh(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sinh(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sinh(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_SINH_, 0);
#ifndef K_ENABLED_SINH_
    fpga_cpuemu_sinh(A, B);
#else
    printf("añadir tensor B\n"); exi(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_sinh_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sinh_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sinh_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_SINH_, 1);
}

// -----------------------------------------------------------------
// sqr
//
void fpga_cpuemu_sqr(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sqr(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sqr(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_SQR_, 0);
#ifndef K_ENABLED_SQR_
    fpga_cpuemu_sqr(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_sqr_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sqr_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sqr_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_SQR_, 1);
}

// -----------------------------------------------------------------
// sqrt
//
void fpga_cpuemu_sqrt(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sqrt(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sqrt(Tensor *A, Tensor *B) {
    _profile_fpga(_FPGA_SQRT_, 0);
#ifndef K_ENABLED_SQRT_
    fpga_cpuemu_sqrt(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_sqrt_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sqrt_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sqrt_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_SQRT_, 1);
}

// -----------------------------------------------------------------
// tan
//
void fpga_cpuemu_tan(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_tan(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_tan(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_TAN_, 0);
#ifndef K_ENABLED_TAN_
    fpga_cpuemu_tan(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_tan_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_tan_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_tan_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_TAN_, 1);
}

// -----------------------------------------------------------------
// tanh
//
void fpga_cpuemu_tanh(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_tanh(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_tanh(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_TANH_, 0);
#ifndef K_ENABLED_TANH_
    fpga_cpuemu_tanh(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_tanh_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_tanh_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_tanh_, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_TANH_, 1);
}

// -----------------------------------------------------------------
// trunc_
//
void fpga_cpuemu_trunc(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_trunc(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_trunc(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_TRUNC_, 0);
#ifndef K_ENABLED_TRUNC_
    fpga_cpuemu_trunc(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_trunc_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_trunc_.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_trunc_, NULL, &event));
    q.finish();
#endif
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
#ifndef K_ENALBED_ADD
    fpga_cpuemu_add(scA, A, scB, B, C, incC);
#else
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_add.setArg(0, scA));
    OCL_CHECK(err, err = kernel_add.setArg(1, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add.setArg(2, scB));
    OCL_CHECK(err, err = kernel_add.setArg(3, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add.setArg(4, *(C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_add.setArg(5, incC));
    OCL_CHECK(err, err = kernel_add.setArg(6, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_add, NULL, &event));
    q.finish();
#endif
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
#ifndef K_ENABLED_INC
  fpga_cpuemu_inc(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_inc.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_inc.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_inc.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_inc, NULL, &event));
  q.finish();
#endif
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
    _profile_fpga_tensor(A);
    _profile_fpga_tensor(B);
    _profile_fpga_tensor(C);
#ifndef K_ENABLED_MULT2D
    fpga_cpuemu_mult2D(A, tA, B, tB, C, incC);
#else
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_mult2d.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_mult2d.setArg(1, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_mult2d.setArg(2, *(C->fpga_ptr)));
    OCL_CHECK(err, err = kernel_mult2d.setArg(3, (int)A->shape[0]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(4, (int)A->shape[1]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(5, (int)B->shape[0]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(6, (int)B->shape[1]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(7, tA));
    OCL_CHECK(err, err = kernel_mult2d.setArg(8, tB));
    OCL_CHECK(err, err = kernel_mult2d.setArg(9, incC));

    OCL_CHECK(err, err = q.enqueueTask(kernel_mult2d, NULL, &event));
    q.finish();

#endif
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
  _profile_fpga_tensor(A);
  _profile_fpga_tensor(B);
  _profile_fpga_tensor(C);
#ifndef K_ENABLED_EL_DIV
  fpga_cpuemu_el_div(A, B, C, incC);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_el_div.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_el_div.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_el_div.setArg(2, *(C->fpga_ptr)));
  OCL_CHECK(err, err = kernel_el_div.setArg(3, incC));
  OCL_CHECK(err, err = kernel_el_div.setArg(4, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_el_div, NULL, &event))
  q.finish();
#endif
  _profile_fpga(_FPGA_EL_DIV, 1);
  _profile_fpga_tensor(A);
  _profile_fpga_tensor(B);
  _profile_fpga_tensor(C);
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
#ifndef K_ENABLED_EL_MULT
  fpga_cpuemu_el_mult(A, B, C, incC);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_el_mult.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_el_mult.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_el_mult.setArg(2, *(C->fpga_ptr)));
  OCL_CHECK(err, err = kernel_el_mult.setArg(3, incC));
  OCL_CHECK(err, err = kernel_el_mult.setArg(4, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_el_mult, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_EL_MULT, 1);
}

// -----------------------------------------------------------------
// sign
//
void fpga_cpuemu_sign(Tensor *A, Tensor *B, float zero_sign) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_sign(A, B, zero_sign);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_sign(Tensor *A, Tensor *B, float zero_sign){
  _profile_fpga(_FPGA_SIGN2, 0);
#ifndef K_ENABLED_SIGN2
  fpga_cpuemu_sign(A, B, zero_sign);
#else
  printf("añadir zero sign\n"); exit(1);
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sign2.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sign2.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sign2.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sign2, NULL, &event));
  q.finish();
#endif
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
#ifndef K_ENABLED_SUM2D_ROWWISE
  fpga_cpuemu_sum2D_rowwise(A, B, C);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(2, *(C->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(3, A->shape[0]));
  OCL_CHECK(err, err = kernel_sum2D_rowwise.setArg(4, A->shape[1]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sum2D_rowwise, NULL, &event));
  q.finish();
#endif
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
#ifndef K_ENABLED_SUM2D_COLWISE
  fpga_cpuemu_sum2D_colwise(A, B, C);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(2, *(C->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(3, A->shape[0]));
  OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(4, A->shape[1]));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sum2D_rowwise, NULL, &event));
  q.finish();
#endif
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
#ifndef K_ENABLED_MAX
  ret = fpga_cpuemu_max(A);
#else
  printf("fpga_max not implemented yet\n"); exit(1);
  // cl_int err;
  // cl::Event event;
  //
  // OCL_CHECK(err, err = kernel_max.setArg(0, *(A->fpga_ptr)));
  // OCL_CHECK(err, err = kernel_max.setArg(1, (long int)A->size));
  //
  // OCL_CHECK(err, err = q.enqueueTask(kernel_max, NULL, &event));
  // q.finish();
#endif
  _profile_fpga(_FPGA_MAX, 1);
  printf("please revise return in cpu_max and compare with fpga_max\n");
  return ret;
}

void fpga_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
	printf("fpga_max not implemented yet\n"); exit(1);
}

int fpga_argmax(Tensor *A) {printf("fpga_argmax not implemented yet\n"); exit(1);}


void fpga_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){printf("fpga_argmax not implemented yet\n"); exit(1);}

std::tuple<float, int> fpga_max(float *ptr, int size, int *map) {printf("fpga_max not implemented yet\n"); exit(1);}


// -------------------------------------------------------------------
// fpga_min
//

std::tuple<float, int> fpga_min(float *ptr, int size, int *map) {
  printf("fpga_min(...) not implemented yet\n");
  exit(1);
}

float fpga_min(Tensor *A) {printf("fpga_min not implemented yet\n"); exit(1);}


void fpga_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){printf("fpga_min not implemented yet\n"); exit(1);}

int fpga_argmin(Tensor *A) {
    auto t = fpga_min(A->ptr, A->size, nullptr);
    return std::get<1>(t);  // get argmin
}


void fpga_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
  printf("fpga_argmin not implemented yet\n");
  exit(1);
}


// -----------------------------------------------------------------
// fpga_sum
//
//
float fpga_sum(Tensor *A) {printf("fpga_sum not implemented yet\n"); exit(1);}

void fpga_sum(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){printf("fpga_sum not implemented yet\n"); exit(1);}

float fpga_sum(float *ptr, int size, int *map) {printf("fpga_sum not implemented yet\n"); exit(1);}

float fpga_sum_abs(Tensor *A) {
	printf("fpga_sum_abs not implemented yet\n"); exit(1);
}


void fpga_sum_abs(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
	printf("fpga_sum_abs not implemented yet\n"); exit(1);
}

float fpga_sum_abs(float *ptr, int size, int *map) {
	printf("fpga_sum_abs not implemented yet\n"); exit(1);
}

float fpga_prod(Tensor *A) {
	printf("fpga_prod not implemented yet\n"); exit(1);
}


void fpga_prod(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
	printf("fpga_prod not implemented yet\n"); exit(1);
}

float fpga_prod(float *ptr, int size, int *map) {
	printf("fpga_prod not implemented yet\n"); exit(1);
}


int fpga_mode(Tensor *A) {
  printf("fpga_mode not implemented yet\n"); exit(1);
    return fpga_mode(A->ptr, A->size, nullptr);
}


void fpga_mode(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
  printf("fpga_mode not implmented yet\n"); exit(1);
}

int fpga_mode(float *ptr, int size, int *map) {
  printf("fpga_mode not implemented yet\n"); exit(1);
}

float fpga_mean(Tensor *A) {printf("fpga_mean not implemented yet\n"); exit(1);}

void fpga_mean(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){printf("fpga_mean not implemented yet\n"); exit(1);}

float fpga_var(Tensor *A, bool unbiased){
	printf("fpga_var not implemented yet\n"); exit(1);
}


void fpga_var(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){
	printf("fpga_var not implemented yet\n"); exit(1);
}

float fpga_var(float *ptr, int size, int *map, bool unbiased){
	printf("fpga_var not implemented yet\n"); exit(1);
}

float fpga_std(Tensor *A, bool unbiased) {printf("fpga_std not implemented yet\n"); exit(1);}

void fpga_std(Tensor *A, Tensor *B, ReduceDescriptor2 *rd, bool unbiased){printf("fpga_std not implemented yet\n"); exit(1);}

float fpga_median(Tensor *A) {
	printf("fpga_median not implemented yet\n"); exit(1);
}


void fpga_median(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
	printf("fpga_median not implemented yet\n"); exit(1);
}

float fpga_median(float *ptr, int size, int *map) {
	printf("fpga_median not implemented yet\n"); exit(1);
}

void fpga_maximum(Tensor* A, Tensor* B, float v){
	printf("fpga_maximum not implemented yet\n"); exit(1);
}

void fpga_maximum(Tensor* A, Tensor* B, Tensor* C){
	printf("fpga_maximum not implemented yet\n"); exit(1);
}

void fpga_minimum(Tensor* A, Tensor* B, float v){
printf("fpga_minimum not implemented yet\n"); exit(1);
}

void fpga_minimum(Tensor* A, Tensor* B, Tensor* C){
	printf("fpga_minimum not implemented yet\n"); exit(1);
}

