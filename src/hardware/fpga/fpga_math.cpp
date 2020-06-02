/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/



#include "eddl/hardware/fpga/fpga_hw.h"

// CPU: Math (in-place) ********************************************

void fpga_abs_(Tensor *A) {
    _profile_fpga(_FPGA_ABS_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ABS_, 1);
}

void fpga_acos_(Tensor *A){
    _profile_fpga(_FPGA_ACOS_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ACOS_, 1);
}

void fpga_add_(Tensor *A, float v) {
    _profile_fpga(_FPGA_ADD_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ADD_, 1);
}

void fpga_asin_(Tensor *A){
    _profile_fpga(_FPGA_ASIN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ASIN_, 1);
}

void fpga_atan_(Tensor *A){
    _profile_fpga(_FPGA_ATAN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ATAN_, 1);
}

void fpga_ceil_(Tensor *A){
    _profile_fpga(_FPGA_CEIL_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_CEIL_, 1);
}

void fpga_clamp_(Tensor *A, float min, float max){
    _profile_fpga(_FPGA_CLAMP_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_CLAMP_, 1);
}


void fpga_cos_(Tensor *A){
    _profile_fpga(_FPGA_COS_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_COS_, 1);
}

void fpga_cosh_(Tensor *A){
    _profile_fpga(_FPGA_COSH_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_COSH_, 1);
}

void fpga_exp_(Tensor *A) {
    _profile_fpga(_FPGA_EXP_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_EXP_, 1);
}

void fpga_floor_(Tensor *A){
    _profile_fpga(_FPGA_FLOOR_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_FLOOR_, 1);
}

void fpga_inv_(Tensor *A, float v){
    _profile_fpga(_FPGA_INV_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_INV_, 1);
}

void fpga_log_(Tensor *A) {
    _profile_fpga(_FPGA_LOG_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOG_, 1);
}

void fpga_log2_(Tensor *A) {
    _profile_fpga(_FPGA_LOG2_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOG2_, 1);
}

void fpga_log10_(Tensor *A) {
    _profile_fpga(_FPGA_LOG10_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOG10_, 1);
}

void fpga_logn_(Tensor *A, float n) {
    _profile_fpga(_FPGA_LOGN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_LOGN_, 1);
}

void fpga_mod_(Tensor *A, float v){
    _profile_fpga(_FPGA_MOD_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_MOD_, 1);
}

void fpga_mult_(Tensor *A, float v) {
    _profile_fpga(_FPGA_MULT_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_MULT_, 1);
}

void fpga_normalize_(Tensor *A, float min, float max){
    _profile_fpga(_FPGA_NORMALIZE_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_NORMALIZE_, 1);
}

void fpga_pow_(Tensor *A, float exp) {
    _profile_fpga(_FPGA_POW_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_POW_, 1);
}

void fpga_powb_(Tensor *A, float base) {
    _profile_fpga(_FPGA_POWB_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_POWB_, 1);
}

void fpga_reciprocal_(Tensor *A) {
    _profile_fpga(_FPGA_RECIPROCAL_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RECIPROCAL_, 1);
}

void fpga_remainder_(Tensor *A, float v) {
    _profile_fpga(_FPGA_REMAINDER_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_REMAINDER_, 1);
}

void fpga_round_(Tensor *A){
    _profile_fpga(_FPGA_ROUND_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ROUND_, 1);
}

void fpga_rsqrt_(Tensor *A){
    _profile_fpga(_FPGA_RSQRT_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_RSQRT_, 1);
}

void fpga_sigmoid_(Tensor *A){
    _profile_fpga(_FPGA_SIGMOID_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SIGMOID_, 1);
}

void fpga_sign_(Tensor *A){
    _profile_fpga(_FPGA_SIGN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SIGN_, 1);
}


void fpga_sin_(Tensor *A){
    _profile_fpga(_FPGA_SIN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SIN_, 1);
}

void fpga_sinh_(Tensor *A){
    _profile_fpga(_FPGA_SINH_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SINH_, 1);
}

void fpga_sqr_(Tensor *A) {
    _profile_fpga(_FPGA_SQR_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SQR_, 1);
}

void fpga_sqrt_(Tensor *A) {
    _profile_fpga(_FPGA_SQRT_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SQRT_, 1);
}

void fpga_tan_(Tensor *A){
    _profile_fpga(_FPGA_TAN_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_TAN_, 1);
}

void fpga_tanh_(Tensor *A){
    _profile_fpga(_FPGA_TANH_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_TANH_, 1);
}

void fpga_trunc_(Tensor *A){
    _profile_fpga(_FPGA_TRUNC_, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_TRUNC_, 1);
}



// FPGA: Math (static) ***************************

void fpga_add(float scA, Tensor *A, float scB, Tensor *B, Tensor *C, int incC) {
    _profile_fpga(_FPGA_ADD, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_ADD, 1);
}


void fpga_inc(Tensor *A, Tensor *B) {
  _profile_fpga(_FPGA_INC, 0);
  B->tsem->lock();
  printf("fpga_ not implemented yet\n"); exit(1);
  B->tsem->unlock();
    _profile_fpga(_FPGA_INC, 1);
}

void fpga_mult2D(Tensor *A, int tA, Tensor *B, int tB, Tensor *C, int incC) {
    _profile_fpga(_FPGA_MULT2D, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_MULT2D, 1);
}

void fpga_el_div(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile_fpga(_FPGA_EL_DIV, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_EL_DIV, 1);
}


void fpga_el_mult(Tensor *A, Tensor *B, Tensor *C, int incC) {
  _profile_fpga(_FPGA_EL_MULT, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_EL_MULT, 1);
}

void fpga_sign2(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SIGN2, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SIGN2, 1);
   
}

void fpga_sum2D_rowwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile_fpga(_FPGA_SUM2D_ROWWISE, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM2D_ROWWISE, 1);
}

void fpga_sum2D_colwise(Tensor *A, Tensor *B, Tensor *C) {
  _profile_fpga(_FPGA_SUM2D_COLWISE, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM2D_COLWISE, 1);
}

// FPGA: Should be reductions ***************************
float fpga_max(Tensor *A){
  _profile_fpga(_FPGA_MAX, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_MAX, 1);
  return 0;
}

float fpga_min(Tensor *A){
  _profile_fpga(_FPGA_MIN, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_MIN, 1);
  return 0; 
}

float fpga_sum(Tensor *A) {
  _profile_fpga(_FPGA_SUM, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM, 1);
  return 0;
}

float fpga_sum_abs(Tensor *A) {
  _profile_fpga(_FPGA_SUM_ABS, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SUM_ABS, 1);
  return 0;
}
