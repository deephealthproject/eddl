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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_abs_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_abs_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_abs_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_acos_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_acos_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_acos_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_add_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_add_.setArg(1, v));
        OCL_CHECK(err, err = kernel_add_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_add_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_asin_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_asin_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_asin_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_atan_.setArg(0, *(A->fpga_ptr)))
        OCL_CHECK(err, err = kernel_atan_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_atan_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_ceil_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_ceil_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_ceil_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_clamp_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_clamp_.setArg(1, min));
        OCL_CHECK(err, err = kernel_clamp_.setArg(2, max));
        OCL_CHECK(err, err = kernel_clamp_.setArg(3, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_clamp_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_cos_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_cos_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_cos_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_cosh_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_cosh_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_cosh_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_exp_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_exp_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_exp_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_floor_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_floor_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_floor_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_inv_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_inv_.setArg(1, v));
        OCL_CHECK(err, err = kernel_inv_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_inv_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_log_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_log_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_log_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_log2_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_log2_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_log2_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_log10_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_log10_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_log10_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_logn_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_logn_.setArg(1, n));
        OCL_CHECK(err, err = kernel_logn_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_logn_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_mod_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_mod_.setArg(1, v));
        OCL_CHECK(err, err = kernel_mod_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_mod_, NULL, &event));
        q.finish();
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
    _profile_fpga_tensor(A);
#ifndef K_ENABLED_MULT_
    fpga_cpuemu_mult_(A, v);
#else
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_pow_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_pow_.setArg(1, exp));
        OCL_CHECK(err, err = kernel_pow_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_pow_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_powb_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_powb_.setArg(1, base));
        OCL_CHECK(err, err = kernel_powb_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_powb_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_reciprocal_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_reciprocal_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_reciprocal_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_remainder_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_remainder_.setArg(1, v));
        OCL_CHECK(err, err = kernel_remainder_.setArg(2, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_remainder_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_round_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_round_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_round_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_rsqrt_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_rsqrt_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_rsqrt_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_sigmoid_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_sigmoid_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_sigmoid_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_sign_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_sign_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_sign_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_sin_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_sin_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_sin_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_sinh_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_sinh_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_sinh_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_sqr_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_sqr_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_sqr_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_sqrt_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_sqrt_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_sqrt_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_tan_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_tan_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_tan_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_tanh_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_tanh_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_tanh_, NULL, &event));
        q.finish();
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
        cl_int err;
        cl::Event event;

        OCL_CHECK(err, err = kernel_trunc_.setArg(0, *(A->fpga_ptr)));
        OCL_CHECK(err, err = kernel_trunc_.setArg(1, (long int)A->size));

        OCL_CHECK(err, err = q.enqueueTask(kernel_trunc_, NULL, &event));
        q.finish();
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
  if (fpga_set_cpuemu_inc == 1) {
    fpga_cpuemu_inc(A, B);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_inc.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_inc.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_inc.setArg(2, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_inc, NULL, &event));
      q.finish();
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
    OCL_CHECK(err, err = kernel_mult2d.setArg(3, A->shape[0]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(4, A->shape[1]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(5, B->shape[0]));
    OCL_CHECK(err, err = kernel_mult2d.setArg(6, B->shape[1]));
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
  if (fpga_set_cpuemu_el_div == 1) {
    fpga_cpuemu_el_div(A, B, C, incC);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_el_div.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_el_div.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_el_div.setArg(2, *(C->fpga_ptr)));
      OCL_CHECK(err, err = kernel_el_div.setArg(3, incC));
      OCL_CHECK(err, err = kernel_el_div.setArg(4, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_el_div, NULL, &event))
      q.finish();
  }
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
  if (fpga_set_cpuemu_el_mult == 1) {
    fpga_cpuemu_el_mult(A, B, C, incC);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_el_mult.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_el_mult.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_el_mult.setArg(2, *(C->fpga_ptr)));
      OCL_CHECK(err, err = kernel_el_mult.setArg(3, incC));
      OCL_CHECK(err, err = kernel_el_mult.setArg(4, (long int)A->size));

      OCL_CHECK(err, err = q.enqueueTask(kernel_el_mult, NULL, &event));
      q.finish();
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
    cl_int err;
    cl::Event event;

    OCL_CHECK(err, err = kernel_sign2.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sign2.setArg(1, *(B->fpga_ptr)));
    OCL_CHECK(err, err = kernel_sign2.setArg(2, (long int)A->size));

    OCL_CHECK(err, err = q.enqueueTask(kernel_sign2, NULL, &event));
    q.finish();
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
  if (fpga_set_cpuemu_sum2D_colwise == 1) {
      fpga_cpuemu_sum2D_colwise(A, B, C);
  } else {
      cl_int err;
      cl::Event event;

      OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(0, *(A->fpga_ptr)));
      OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(1, *(B->fpga_ptr)));
      OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(2, *(C->fpga_ptr)));
      OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(3, A->shape[0]));
      OCL_CHECK(err, err = kernel_sum2D_colwise.setArg(4, A->shape[1]));

      OCL_CHECK(err, err = q.enqueueTask(kernel_sum2D_rowwise, NULL, &event));
      q.finish();
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
      // cl_int err;
      // cl::Event event;
      //
      // OCL_CHECK(err, err = kernel_max.setArg(0, *(A->fpga_ptr)));
      // OCL_CHECK(err, err = kernel_max.setArg(1, (long int)A->size));
      //
      // OCL_CHECK(err, err = q.enqueueTask(kernel_max, NULL, &event));
      // q.finish();
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
      // cl_int err;
      // cl::Event event;
      //
      // OCL_CHECK(err, err = kernel_min.setArg(0, *(A->fpga_ptr)));
      // OCL_CHECK(err, err = kernel_min.setArg(1, (long int)A->size));
      //
      // OCL_CHECK(err, err = q.enqueueTask(kernel_min, NULL, &event));
      // q.finish();
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
  _profile_fpga_tensor(A);
#ifndef K_ENABLED_SUM
    ret = fpga_cpuemu_sum(A);
#else
    // cl_int err;
    // cl::Event event, result_ready;
    // //cl::Context context;
    //
    // //float sum = (float) malloc(sizeof(float));
    // float *sum = 0;
    // OCL_CHECK(err, cl::Buffer a(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(float) ,sum, &err));
    //
    // OCL_CHECK(err, err = kernel_sum.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_sum.setArg(1, (long int)A->size));
    // OCL_CHECK(err, err = kernel_sum.setArg(2, a));
    // OCL_CHECK(err, err = q.enqueueTask(kernel_sum, NULL, &event));
    // event.wait();
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({a},CL_MIGRATE_MEM_OBJECT_HOST, NULL, &result_ready));
    // result_ready.wait();
    // return *sum;

#endif
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
    // cl_int err;
    // cl::Event event;
    //
    // OCL_CHECK(err, err = kernel_sum_abs.setArg(0, *(A->fpga_ptr)));
    // OCL_CHECK(err, err = kernel_sum_abs.setArg(1, (long int)A->size));
    //
    // OCL_CHECK(err, err = q.enqueueTask(kernel_sum_abs, NULL, &event));
    // q.finish();
  }
  _profile_fpga(_FPGA_SUM_ABS, 1);
  return ret;
}
