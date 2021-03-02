/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

extern cl::Kernel mult2D;
extern cl::Kernel sum2D_rowwise;

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
  _profile_fpga(_FPGA_ABS, 0);
#ifndef K_ENABLED_ABS
  fpga_cpuemu_abs(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_abs.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_abs.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_abs.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_abs, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_ABS, 1);
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
  _profile_fpga(_FPGA_ACOS, 0);
#ifndef K_ENABLED_ACOS
  fpga_cpuemu_acos(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_acos.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_acos.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_acos.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_acos, NULL, &event));
  q.finish();
#endif
    _profile_fpga(_FPGA_ACOS, 1);
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
  _debug_fpga_funcs("add");
  _profile_fpga(_FPGA_ADD, 0);
#ifndef K_ENABLED_ADD
  fpga_cpuemu_add(A, B, v);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_add.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_add.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_add.setArg(2, v));
  OCL_CHECK(err, err = kernel_add.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_add, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_ADD, 1);
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
  _profile_fpga(_FPGA_ASIN, 0);
#ifndef K_ENABLED_ASIN
  fpga_cpuemu_asin(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_asin.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_asin.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_asin.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_asin, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_ASIN, 1);
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
  _profile_fpga(_FPGA_ATAN, 0);
#ifndef K_ENABLED_ATAN
  fpga_cpuemu_atan(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_atan.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_atan.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_atan.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_atan, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_ATAN, 1);
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
  _profile_fpga(_FPGA_CEIL, 0);
#ifndef K_ENABLED_CEIL
  fpga_cpuemu_ceil(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_ceil.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_ceil.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_ceil.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_ceil, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_CEIL, 1);
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
  _profile_fpga(_FPGA_CLAMP, 0);
#ifndef K_ENABLED_CLAMP
  fpga_cpuemu_clamp(A, B, min, max);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_clamp.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_clamp.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_clamp.setArg(2, min));
  OCL_CHECK(err, err = kernel_clamp.setArg(3, max));
  OCL_CHECK(err, err = kernel_clamp.setArg(4, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_clamp, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_CLAMP, 1);
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
  _profile_fpga(_FPGA_COS, 0);
#ifndef K_ENABLED_COS
  fpga_cpuemu_cos(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_cos.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cos.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cos.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_cos, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_COS, 1);
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
  _profile_fpga(_FPGA_COSH, 0);
#ifndef K_ENABLED_COSH
  fpga_cpuemu_cosh(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_cosh.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cosh.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_cosh.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_cosh, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_COSH, 1);
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
  _profile_fpga(_FPGA_EXP, 0);
#ifndef K_ENABLED_EXP
  fpga_cpuemu_exp(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_exp.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_exp.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_exp.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_exp, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_EXP, 1);
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
  _profile_fpga(_FPGA_FLOOR, 0);
#ifndef K_ENABLED_FLOOR
  fpga_cpuemu_floor(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_floor.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_floor.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_floor.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_floor, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_FLOOR, 1);
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
  _profile_fpga(_FPGA_INV, 0);
#ifndef K_ENABLED_INV
  fpga_cpuemu_inv(A, B, v);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_inv.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_inv.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_inv.setArg(2, v));
  OCL_CHECK(err, err = kernel_inv.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_inv, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_INV, 1);
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
  _profile_fpga(_FPGA_LOG, 0);
#ifndef K_ENABLED_LOG
  fpga_cpuemu_log(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_log.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_log.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_log.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_log, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_LOG, 1);
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
    _profile_fpga(_FPGA_LOG2, 0);
#ifndef K_ENABLED_LOG2
    fpga_cpuemu_log2(A, B);
#else
    printf("añadir tensor B\n"); exit(1);
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_log2.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_log2.setArg(1, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_log2, NULL, &event));
    q.finish();
#endif
    _profile_fpga(_FPGA_LOG2, 1);
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
  _profile_fpga(_FPGA_LOG10, 0);
#ifndef K_ENABLED_LOG10
  fpga_cpuemu_log10(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_log10.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_log10.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_log10.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_log10, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_LOG10, 1);
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
  _profile_fpga(_FPGA_LOGN, 0);
#ifndef K_ENABLED_LOGN
  fpga_cpuemu_logn(A, B, n);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_logn.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_logn.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_logn.setArg(2, n));
  OCL_CHECK(err, err = kernel_logn.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_logn, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_LOGN, 1);
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
  _profile_fpga(_FPGA_MOD, 0);
#ifndef K_ENABLED_MOD
  fpga_cpuemu_mod(A, B, v);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_mod.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_mod.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_mod.setArg(2, v));
  OCL_CHECK(err, err = kernel_mod.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_mod, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_MOD, 1);
}

// -----------------------------------------------------------------
// mult
//
void fpga_cpuemu_mult(Tensor *A, Tensor *B, float v) {
  fpga_copy_from_fpga(A, A->ptr, 0);
  fpga_data_type *ptrA = (fpga_data_type *)A->ptr;
  fpga_data_type *ptrB = (fpga_data_type *)B->ptr;
  fpga_data_type vv = (fpga_data_type)v;
  #pragma omp parallel for
  for (int i = 0; i < A->size; ++i) ptrB[i] = ptrA[i] * vv;
  fpga_copy_to_fpga(B->ptr, B, 0);
}

void fpga_mult(Tensor *A, Tensor *B, float v) {
  _debug_fpga_funcs("mult");
  _profile_fpga(_FPGA_MULT, 0);
  _profile_fpga_tensor(A);
#ifndef K_ENABLED_MULT
  fpga_cpuemu_mult(A, B, v);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_mult.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_mult.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_mult.setArg(2, v));
  OCL_CHECK(err, err = kernel_mult.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_mult, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_MULT, 1);
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
  _profile_fpga(_FPGA_NORMALIZE, 0);
#ifndef K_ENABLED_NORMALIZE
  fpga_cpuemu_normalize(A, B, min, max);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_normalize.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_normalize.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_normalize.setArg(2, min));
  OCL_CHECK(err, err = kernel_normalize.setArg(3, max));
  OCL_CHECK(err, err = kernel_normalize.setArg(4, (long int)A->size));
  OCL_CHECK(err, err = kernel_normalize.setArg(5, A->min()));
  OCL_CHECK(err, err = kernel_normalize.setArg(6, A->max()));

  OCL_CHECK(err, err = q.enqueueTask(kernel_normalize, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_NORMALIZE, 1);
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
  _profile_fpga(_FPGA_POW, 0);
#ifndef K_ENABLED_POW
  fpga_cpuemu_pow(A, B, exp);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_pow.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_pow.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_pow.setArg(2, exp));
  OCL_CHECK(err, err = kernel_pow.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_pow, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_POW, 1);
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
  _profile_fpga(_FPGA_POWB, 0);
#ifndef K_ENABLED_POWB
  fpga_cpuemu_powb(A, B, base);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_powb.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_powb.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_powb.setArg(2, base));
  OCL_CHECK(err, err = kernel_powb.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_powb, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_POWB, 1);
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
  _profile_fpga(_FPGA_REMAINDER, 0);
#ifndef K_ENABLED_REMAINDER
  fpga_cpuemu_remainder(A, B, v);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_remainder.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_remainder.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_remainder.setArg(2, v));
  OCL_CHECK(err, err = kernel_remainder.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_remainder, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_REMAINDER, 1);
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
  _profile_fpga(_FPGA_ROUND, 0);
#ifndef K_ENABLED_ROUND
  fpga_cpuemu_round(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_round.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_round.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_round.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_round, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_ROUND, 1);
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
  _profile_fpga(_FPGA_RSQRT, 0);
#ifndef K_ENABLED_RSQRT
  fpga_cpuemu_rsqrt(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_rsqrt.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_rsqrt.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_rsqrt.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_rsqrt, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_RSQRT, 1);
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
  _profile_fpga(_FPGA_SIN, 0);
#ifndef K_ENABLED_SIN
  fpga_cpuemu_sin(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sin.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sin.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sin.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sin, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SIN, 1);
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
  _profile_fpga(_FPGA_SINH, 0);
#ifndef K_ENABLED_SINH
  fpga_cpuemu_sinh(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sinh.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sinh.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sinh.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sinh, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SINH, 1);
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
  _profile_fpga(_FPGA_SQR, 0);
#ifndef K_ENABLED_SQR
  fpga_cpuemu_sqr(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sqr.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sqr.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sqr.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sqr, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SQR, 1);
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
  _profile_fpga(_FPGA_SQRT, 0);
#ifndef K_ENABLED_SQRT
  fpga_cpuemu_sqrt(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sqrt.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sqrt.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sqrt.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sqrt, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SQRT, 1);
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
  _profile_fpga(_FPGA_TAN, 0);
#ifndef K_ENABLED_TAN
  fpga_cpuemu_tan(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_tan.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_tan.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_tan.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_tan, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_TAN, 1);
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
  _profile_fpga(_FPGA_TANH, 0);
#ifndef K_ENABLED_TANH
  fpga_cpuemu_tanh(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_tanh.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_tanh.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_tanh.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_tanh, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_TANH, 1);
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
  _profile_fpga(_FPGA_TRUNC, 0);
#ifndef K_ENABLED_TRUNC
  fpga_cpuemu_trunc(A, B);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_trunc.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_trunc.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_trunc.setArg(2, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_trunc, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_TRUNC, 1);
}

// FPGA: Math (static) ***************************

// -----------------------------------------------------------------
// add
//
void fpga_cpuemu_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
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
  fpga_copy_from_fpga(A, A->ptr);
  cpu_inc(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_inc(Tensor *A, Tensor *B) {
  _profile_fpga(_FPGA_INC, 0);
                 // why locks?
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

  _profile_fpga(_FPGA_INC, 1);
}

// -----------------------------------------------------------------
// mult2D
//
void fpga_cpuemu_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_mult2D(A, tA, B, tB, C, incC);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    _debug_fpga_funcs("mult2D");
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
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_from_fpga(B, B->ptr);
  cpu_el_div(A, B, C, incC);
  fpga_copy_to_fpga(C->ptr, C);
}

void fpga_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _debug_fpga_funcs("el_div");
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
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sign2.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sign2.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sign2.setArg(2, zero_sign));
  OCL_CHECK(err, err = kernel_sign2.setArg(3, (long int)A->size));

  OCL_CHECK(err, err = q.enqueueTask(kernel_sign2, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SIGN2, 1);

}

// -----------------------------------------------------------------
// sum2D_rowwise
//
void fpga_cpuemu_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {

  fpga_copy_from_fpga(A, A->ptr, 0);
  fpga_copy_from_fpga(B, B->ptr, 0);
  fpga_data_type *ptrA = (fpga_data_type *)A->ptr;
  fpga_data_type *ptrB = (fpga_data_type *)B->ptr;
  fpga_data_type *ptrC = (fpga_data_type *)C->ptr;

  #pragma omp parallel for
  for (int i = 0; i < A->shape[0]; i++) {
      int p=i*A->shape[1];
      for (int j = 0; j < A->shape[1]; j++, p++)
        ptrC[p] = ptrA[p] + ptrB[j];
  }
  fpga_copy_to_fpga(C->ptr, C, 0);
}

void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  _debug_fpga_funcs("sum2D_rowwise");
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
// max
//

float fpga_cpuemu_max(Tensor *A) {
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
  cl_int err;
  cl::Event event;
  
  OCL_CHECK(err, err = kernel_max.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_max.setArg(1, (long int)A->size));
  // Jorge, añade parametro de return 
  OCL_CHECK(err, err = q.enqueueTask(kernel_max, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_MAX, 1);
  return ret;
}

// -------------------------------------------------------------------
// max
//
void fpga_cpuemu_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(rd->fpga_index, (void *)&rd->index, rd->index.size());
  // index[i].data must be read from fpga
  printf("Not properly implemented yet (fpga_cpuemu_max\n"); exit(1);
  cpu_max(A, B, rd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_max(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
  _profile_fpga(_FPGA_MAX, 0);
#ifndef K_ENABLED_MAX_2
  fpga_cpuemu_max(A, B, rd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_max_2.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_max_2.setArg(1, *(B->fpga_ptr)));
  // añadir mas parametros
  printf("fpga_max not implemented yet\n"); exit(1);
  OCL_CHECK(err, err = q.enqueueTask(kernel_max_2, NULL, &event));
  q.finish();
#endif
}

// -----------------------------------------------------------------
// argmax
//

int fpga_cpuemu_argmax(Tensor *A) {
  fpga_copy_from_fpga(A, A->ptr);
  int ret = cpu_argmax(A);
  return ret;
}

int fpga_argmax(Tensor *A){
  int ret;
  printf("fpga_argmax\n");
#ifndef K_ENABLED_ARGMAX
  ret = fpga_cpuemu_argmax(A);
#else
  printf("fpga_max not implemented yet\n"); exit(1);
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_argmax.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_argmax.setArg(1, (long int)A->size));
  // Jorge, añade parametro de return 
  OCL_CHECK(err, err = q.enqueueTask(kernel_argmax, NULL, &event));
  q.finish();
#endif
  return ret;
}

// -------------------------------------------------------------------
// argmax
//
void fpga_cpuemu_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(rd->fpga_index, (void *)&rd->index, rd->index.size());
  // index[i].data must be read from fpga
  printf("Not properly implemented yet (fpga_cpuemu_argmax\n"); exit(1);
  cpu_argmax(A, B, rd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_argmax(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
	printf("fpga_argmax\n");
#ifndef K_ENABLED_ARGMAX_2
  fpga_cpuemu_argmax(A, B, rd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_argmax_2.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_argmax_2.setArg(1, *(B->fpga_ptr)));
  // añadir mas parametros
  printf("fpga_max not implemented yet\n"); exit(1);
  OCL_CHECK(err, err = q.enqueueTask(kernel_argmax_2, NULL, &event));
  q.finish();
#endif
}

// -------------------------------------------------------------------
// min
//

float fpga_cpuemu_min(Tensor *A) {
  fpga_copy_from_fpga(A, A->ptr);
  float ret = cpu_min(A);
  return ret;
}

float fpga_min(Tensor *A){
  float ret;
  _profile_fpga(_FPGA_MIN, 0);
#ifndef K_ENABLED_MIN
  ret = fpga_cpuemu_min(A);
#else
  printf("fpga_min not implemented yet\n"); exit(1);
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_min.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_min.setArg(1, (long int)A->size));
  // Jorge, añade parametro de return
  OCL_CHECK(err, err = q.enqueueTask(kernel_min, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_MIN, 1);
  return ret;
}

// -------------------------------------------------------------------
// min
//
void fpga_cpuemu_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(rd->fpga_index, (void *)&rd->index, rd->index.size());
  // index[i].data must be read from fpga
  printf("Not properly implemented yet (fpga_cpuemu_min\n"); exit(1);
  cpu_min(A, B, rd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_min(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
  _profile_fpga(_FPGA_MIN, 0);
#ifndef K_ENABLED_MIN_2
  fpga_cpuemu_min(A, B, rd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_min_2.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_min_2.setArg(1, *(B->fpga_ptr)));
  // añadir mas parametros
  printf("fpga_min not implemented yet\n"); exit(1);
  OCL_CHECK(err, err = q.enqueueTask(kernel_min_2, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_MIN, 1);
}

// -----------------------------------------------------------------
// argmin
//

int fpga_cpuemu_argmin(Tensor *A) {
  fpga_copy_from_fpga(A, A->ptr);
  int ret = cpu_argmin(A);
  return ret;
}

int fpga_argmin(Tensor *A){
  int ret;
  printf("argmin\n");
#ifndef K_ENABLED_ARGMIN
  ret = fpga_cpuemu_argmin(A);
#else
  printf("fpga_min not implemented yet\n"); exit(1);
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_argmin.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_argmin.setArg(1, (long int)A->size));
  // Jorge, añade parametro de return
  OCL_CHECK(err, err = q.enqueueTask(kernel_argmin, NULL, &event));
  q.finish();
#endif
  return ret;
}

// -------------------------------------------------------------------
// argmin
//
void fpga_cpuemu_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd) {
  fpga_copy_from_fpga(A, A->ptr);
  fpga_copy_memory_from_fpga(rd->fpga_index, (void *)&rd->index, rd->index.size());
  // index[i].data must be read from fpga
  printf("Not properly implemented yet (fpga_cpuemu_argmin\n"); exit(1);
  cpu_argmin(A, B, rd);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_argmin(Tensor *A, Tensor *B, ReduceDescriptor2 *rd){
	printf("argmin\n");
#ifndef K_ENABLED_ARGMIN_2
  fpga_cpuemu_argmin(A, B, rd);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_argmin_2.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_argmin_2.setArg(1, *(B->fpga_ptr)));
  // añadir mas parametros
  printf("fpga_min not implemented yet\n"); exit(1);
  OCL_CHECK(err, err = q.enqueueTask(kernel_argmin_2, NULL, &event));
  q.finish();
#endif
}

// -----------------------------------------------------------------
// fpga_sum
//
//
float fpga_cpuemu_sum(Tensor *A) {
  fpga_copy_from_fpga(A, A->ptr);
  float ret = cpu_sum(A);
  return ret;
}

float fpga_sum(Tensor *A) {
  float ret;
  _profile_fpga(_FPGA_SUM, 0);
  _profile_fpga_tensor(A);
#ifndef K_ENABLED_SUM
  ret = fpga_cpuemu_sum(A);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_sum.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_sum.setArg(1, A->size));
  printf("Error, fpga_sum not properly implemented yet\n");
  exit(1);

  OCL_CHECK(err, err = q.enqueueTask(kernel_sum, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SUM, 1);
  return ret;
}

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

#endif
