/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#include "eddl/hardware/fpga/fpga_hw.h"

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
    printf("fpga_cpuemu_abs_ not implemented yet\n");
    exit(1);
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
    printf("fpga_cpuemu_acos_ not implemented yet\n");
    exit(1);
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
void fpga_add_(Tensor *A, float v) {
    _profile_fpga(_FPGA_ADD_, 0);
    
    printf("fpga_add not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ADD_, 1);
}

// -----------------------------------------------------------------
// asin_
//
void fpga_asin_(Tensor *A){
    _profile_fpga(_FPGA_ASIN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ASIN_, 1);
}

// -----------------------------------------------------------------
// atan_
//
void fpga_atan_(Tensor *A){
    _profile_fpga(_FPGA_ATAN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ATAN_, 1);
}

// -----------------------------------------------------------------
// ceil_
//
void fpga_ceil_(Tensor *A){
    _profile_fpga(_FPGA_CEIL_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_CEIL_, 1);
}

// -----------------------------------------------------------------
// clamp_
//
void fpga_clamp_(Tensor *A, float min, float max){
    _profile_fpga(_FPGA_CLAMP_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_CLAMP_, 1);
}

// -----------------------------------------------------------------
// cos_
//
void fpga_cos_(Tensor *A){
    _profile_fpga(_FPGA_COS_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_COS_, 1);
}

// -----------------------------------------------------------------
// cosh_
//
void fpga_cosh_(Tensor *A){
    _profile_fpga(_FPGA_COSH_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_COSH_, 1);
}

// -----------------------------------------------------------------
// exp_
//
void fpga_exp_(Tensor *A) {
    _profile_fpga(_FPGA_EXP_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_EXP_, 1);
}

// -----------------------------------------------------------------
// floor_
//
void fpga_floor_(Tensor *A){
    _profile_fpga(_FPGA_FLOOR_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_FLOOR_, 1);
}

// -----------------------------------------------------------------
// inv_
//
void fpga_inv_(Tensor *A, float v){
    _profile_fpga(_FPGA_INV_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_INV_, 1);
}

// -----------------------------------------------------------------
// log_
//
void fpga_log_(Tensor *A) {
    _profile_fpga(_FPGA_LOG_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOG_, 1);
}

// -----------------------------------------------------------------
// log2_
//
void fpga_log2_(Tensor *A) {
    _profile_fpga(_FPGA_LOG2_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOG2_, 1);
}

// -----------------------------------------------------------------
// log10_
//
void fpga_log10_(Tensor *A) {
    _profile_fpga(_FPGA_LOG10_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOG10_, 1);
}

// -----------------------------------------------------------------
// logn_
//
void fpga_logn_(Tensor *A, float n) {
    _profile_fpga(_FPGA_LOGN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGN_, 1);
}

// -----------------------------------------------------------------
// mod_
//
void fpga_mod_(Tensor *A, float v){
    _profile_fpga(_FPGA_MOD_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_MOD_, 1);
}

// -----------------------------------------------------------------
// mult_
//
void fpga_mult_(Tensor *A, float v) {
    _profile_fpga(_FPGA_MULT_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_MULT_, 1);
}

// -----------------------------------------------------------------
// normalize_
//
void fpga_normalize_(Tensor *A, float min, float max){
    _profile_fpga(_FPGA_NORMALIZE_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_NORMALIZE_, 1);
}

// -----------------------------------------------------------------
// pow_
//
void fpga_pow_(Tensor *A, float exp) {
    _profile_fpga(_FPGA_POW_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_POW_, 1);
}

// -----------------------------------------------------------------
// powb_
//
void fpga_powb_(Tensor *A, float base) {
    _profile_fpga(_FPGA_POWB_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_POWB_, 1);
}

// -----------------------------------------------------------------
// reciprocal_
//
void fpga_reciprocal_(Tensor *A) {
    _profile_fpga(_FPGA_RECIPROCAL_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RECIPROCAL_, 1);
}

// -----------------------------------------------------------------
// remainder_
//
void fpga_remainder_(Tensor *A, float v) {
    _profile_fpga(_FPGA_REMAINDER_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_REMAINDER_, 1);
}

// -----------------------------------------------------------------
// round_
//
void fpga_round_(Tensor *A){
    _profile_fpga(_FPGA_ROUND_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ROUND_, 1);
}

// -----------------------------------------------------------------
// rsqrt_
//
void fpga_rsqrt_(Tensor *A){
    _profile_fpga(_FPGA_RSQRT_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RSQRT_, 1);
}

// -----------------------------------------------------------------
// sigmoid_
//
void fpga_sigmoid_(Tensor *A){
    _profile_fpga(_FPGA_SIGMOID_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SIGMOID_, 1);
}

// -----------------------------------------------------------------
// sign_
//
void fpga_sign_(Tensor *A){
    _profile_fpga(_FPGA_SIGN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SIGN_, 1);
}

// -----------------------------------------------------------------
// sin_
//
void fpga_sin_(Tensor *A){
    _profile_fpga(_FPGA_SIN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SIN_, 1);
}

// -----------------------------------------------------------------
// sinh_
//
void fpga_sinh_(Tensor *A){
    _profile_fpga(_FPGA_SINH_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SINH_, 1);
}

// -----------------------------------------------------------------
// sqr_
//
void fpga_sqr_(Tensor *A) {
    _profile_fpga(_FPGA_SQR_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SQR_, 1);
}

// -----------------------------------------------------------------
// sqrt_
//
void fpga_sqrt_(Tensor *A) {
    _profile_fpga(_FPGA_SQRT_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SQRT_, 1);
}

// -----------------------------------------------------------------
// tan_
//
void fpga_tan_(Tensor *A){
    _profile_fpga(_FPGA_TAN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_TAN_, 1);
}

// -----------------------------------------------------------------
// tanh_
//
void fpga_tanh_(Tensor *A){
    _profile_fpga(_FPGA_TANH_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_TANH_, 1);
}

// -----------------------------------------------------------------
// trunc_
//
void fpga_trunc_(Tensor *A){
    _profile_fpga(_FPGA_TRUNC_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_TRUNC_, 1);
}

// FPGA: Math (static) ***************************

// -----------------------------------------------------------------
// add
//
void fpga_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
    _profile_fpga(_FPGA_ADD, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ADD, 1);
}

// -----------------------------------------------------------------
// inc
//
void fpga_inc(Tensor *A, Tensor *B) {
  _profile_fpga(_FPGA_INC, 0);
  B->tsem->lock();
  printf("fpga_ not implemented yet\n"); exit(1);
  B->tsem->unlock();
    _profile_fpga(_FPGA_INC, 1);
}

// -----------------------------------------------------------------
// mult2D
//
void fpga_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    _profile_fpga(_FPGA_MULT2D, 0);

    cl_int err;
    cl::Event event;
    /*if (incC != 0) {
       printf("WARNING:: Mult2D with inc not supported\n");
    }else printf("REGULAR MATMULT\n");*/

    OCL_CHECK(err, err = mult2D.setArg(0, (A->fpga_ptr)));
    OCL_CHECK(err, err = mult2D.setArg(1, (B->fpga_ptr)));
    OCL_CHECK(err, err = mult2D.setArg(2, (C->fpga_ptr)));
    OCL_CHECK(err, err = mult2D.setArg(3, A->shape[0]));
    OCL_CHECK(err, err = mult2D.setArg(4, A->shape[1]));
    OCL_CHECK(err, err = mult2D.setArg(5, B->shape[0]));
    OCL_CHECK(err, err = mult2D.setArg(6, B->shape[1]));
    OCL_CHECK(err, err = mult2D.setArg(7, tA));
    OCL_CHECK(err, err = mult2D.setArg(8, tB));

    //printf("sizes A(%dx%d) B(%dx%d) C(%dx%d)\n", A->sizes[0], A->sizes[1], B->sizes[0], B->sizes[1],C->sizes[0], C->sizes[1]);
    //OCL_CHECK(err, err = tensor_op.setArg(5, incC));
    OCL_CHECK(err, err = q.enqueueTask(mult2D, NULL, &event));
    //   event.wait();
    q.finish();

    _profile_fpga(_FPGA_MULT2D, 1);
}

// -----------------------------------------------------------------
// el_div
//
void fpga_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile_fpga(_FPGA_EL_DIV, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_EL_DIV, 1);
}

// -----------------------------------------------------------------
// el_mult
//
void fpga_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile_fpga(_FPGA_EL_MULT, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_EL_MULT, 1);
}

// -----------------------------------------------------------------
// sign2
//
void fpga_sign2(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SIGN2, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SIGN2, 1);
   
}

// -----------------------------------------------------------------
// sum2D_rowwise
//
void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile_fpga(_FPGA_SUM2D_ROWWISE, 0);

  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = sum2D_rowwise.setArg(0, (A->fpga_ptr)));
  OCL_CHECK(err, err = sum2D_rowwise.setArg(1, (B->fpga_ptr)));
  OCL_CHECK(err, err = sum2D_rowwise.setArg(2, (C->fpga_ptr)));
  OCL_CHECK(err, err = sum2D_rowwise.setArg(3, A->shape[0]));
  OCL_CHECK(err, err = sum2D_rowwise.setArg(4, A->shape[1]));

  OCL_CHECK(err, err = q.enqueueTask(sum2D_rowwise, NULL, &event));
  q.finish();
  
  _profile_fpga(_FPGA_SUM2D_ROWWISE, 1);
}

// -----------------------------------------------------------------
// sum2D_colwise
//
void fpga_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile_fpga(_FPGA_SUM2D_COLWISE, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM2D_COLWISE, 1);
}

// FPGA: Should be reductions ***************************

// -----------------------------------------------------------------
// fpga_max
//
float fpga_max(Tensor *A){
  _profile_fpga(_FPGA_MAX, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_MAX, 1);
  return 0;
}

// -----------------------------------------------------------------
// fpga_min
//
float fpga_min(Tensor *A){
  _profile_fpga(_FPGA_MIN, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_MIN, 1);
  return 0; 
}

// -----------------------------------------------------------------
// fpga_sum
//
float fpga_sum(Tensor *A) {
  _profile_fpga(_FPGA_SUM, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM, 1);
  return 0;
}

// -----------------------------------------------------------------
// sum_abs
//
float fpga_sum_abs(Tensor *A) {
  _profile_fpga(_FPGA_SUM_ABS, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM_ABS, 1);
  return 0;
}
