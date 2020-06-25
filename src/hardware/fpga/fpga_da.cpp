/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/

#ifdef cFPGA

#include <iostream>
#include <utility>
#include <cmath>

#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/random.h"
#include "eddl/hardware/cpu/cpu_tensor.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// CPU: Data augmentation (2D Optimized) ********************************************

// -----------------------------------------------------------------
// shift
//
void fpga_cpuemu_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_shift(A, B, shift, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant) {
  _profile_fpga(_FPGA_SHIFT, 0);
#ifndef K_ENABLED_SHIFT
  fpga_cpuemu_shift(A, B, shift, mode, constant);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_shift.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_shift.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_shift.setArg(2, A->shape[0]));
  OCL_CHECK(err, err = kernel_shift.setArg(3, A->shape[1]));
  OCL_CHECK(err, err = kernel_shift.setArg(4, A->shape[2]));
  OCL_CHECK(err, err = kernel_shift.setArg(5, A->shape[3]));
  OCL_CHECK(err, err = kernel_shift.setArg(6, B->shape[0]));
  OCL_CHECK(err, err = kernel_shift.setArg(7, B->shape[1]));
  OCL_CHECK(err, err = kernel_shift.setArg(8, B->shape[2]));
  OCL_CHECK(err, err = kernel_shift.setArg(9, B->shape[3]));
  OCL_CHECK(err, err = kernel_shift.setArg(10, A->stride[0]));
  OCL_CHECK(err, err = kernel_shift.setArg(11, A->stride[1]));
  OCL_CHECK(err, err = kernel_shift.setArg(12, A->stride[2]));
  OCL_CHECK(err, err = kernel_shift.setArg(13, A->stride[3]));
  OCL_CHECK(err, err = kernel_shift.setArg(14, B->stride[0]));
  OCL_CHECK(err, err = kernel_shift.setArg(15, B->stride[1]));
  OCL_CHECK(err, err = kernel_shift.setArg(16, B->stride[2]));
  OCL_CHECK(err, err = kernel_shift.setArg(17, B->stride[3]));
  OCL_CHECK(err, err = kernel_shift.setArg(18, shift[0]));
  OCL_CHECK(err, err = kernel_shift.setArg(19, shift[1]));
  OCL_CHECK(err, err = kernel_shift.setArg(20, mode));
  OCL_CHECK(err, err = kernel_shift.setArg(21, constant));

  OCL_CHECK(err, err = q.enqueueTask(kernel_shift, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_SHIFT, 1);
}

// -----------------------------------------------------------------
// rotate
//
void fpga_cpuemu_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_rotate(A, B, angle, offset_center, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant){
	printf("fpga_rotate\n");
  _profile_fpga(_FPGA_ROTATE, 0);
#ifndef K_ENABLED_ROTATE
  fpga_cpuemu_rotate(A, B, angle, offset_center, mode, constant);
#else
#endif
  _profile_fpga(_FPGA_ROTATE, 1);
}

// -----------------------------------------------------------------
// scale
//
void fpga_cpuemu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_scale(A, B, new_shape, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant){
	printf("fpga_scale\n");
  _profile_fpga(_FPGA_SCALE, 0);
#ifndef K_ENABLED_SCALE
  fpga_cpuemu_scale(A, B, new_shape, mode, constant);
#else
  printf("fpga_scale ate not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_SCALE, 1);
}

// -----------------------------------------------------------------
// flip
//
void fpga_cpuemu_flip(Tensor *A, Tensor *B, int axis){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_flip(A, B, axis);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_flip(Tensor *A, Tensor *B, int axis){
  _profile_fpga(_FPGA_FLIP, 0);
#ifndef K_ENABLED_FLIP
  fpga_cpuemu_flip(A, B, axis);
#else
  printf("fpga_flip ate not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_FLIP, 1);
}

// -----------------------------------------------------------------
// crop
//
void fpga_cpuemu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_crop(A, B, coords_from, coords_to, constant, inverse);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
  _profile_fpga(_FPGA_CROP, 0);
#ifndef K_ENABLED_CROP
  fpga_cpuemu_crop(A, B, coords_from, coords_to, constant, inverse);
#else
  printf("fpga_crop ate not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_CROP, 1);
}

// -----------------------------------------------------------------
// crop-scale
//
void fpga_cpuemu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_crop_scale(A, B, coords_from, coords_to, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant){
  _profile_fpga(_FPGA_CROP_SCALE, 0);
#ifndef K_ENABLED_CROP_SCALE
  fpga_cpuemu_crop_scale(A, B, coords_from, coords_to, mode, constant);
#else
  printf("fpga_crop_scale ate not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_CROP_SCALE, 1);
}


// FPGA: Data augmentation (2D Optimized) ********************************************

// -----------------------------------------------------------------
// shift_random
//
void fpga_cpuemu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_shift_random(A, B, factor_x, factor_y, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant) {
  _profile_fpga(_FPGA_SHIFT_RANDOM, 0);
#ifndef K_ENABLED_SHIFT_RANDOM
  fpga_cpuemu_shift_random(A, B, factor_x, factor_y, mode, constant);
#else
  printf("fpga_shift_random is not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_SHIFT_RANDOM, 1);
}

// -----------------------------------------------------------------
// rotate_random
//
void fpga_cpuemu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_rotate_random(A, B, factor, offset_center, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant){
  _profile_fpga(_FPGA_ROTATE_RANDOM, 0);
#ifndef K_ENABLED_ROTATE_RANDOM
  fpga_cpuemu_rotate_random(A, B, factor, offset_center, mode, constant);
#else
  printf("fpga_rotate_random is not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_ROTATE_RANDOM, 1);
}


// -----------------------------------------------------------------
// scale_random
//
void fpga_cpuemu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_scale_random(A, B, factor, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
  _profile_fpga(_FPGA_SCALE_RANDOM, 0);
#ifndef K_ENABLED_SCALE_RANDOM
  fpga_cpuemu_scale_random(A, B, factor, mode, constant);
#else
  printf("fpga_scale_random is not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_SCALE_RANDOM, 1);
}

// -----------------------------------------------------------------
// flip_random
//
void fpga_cpuemu_flip_random(Tensor *A, Tensor *B, int axis) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_flip_random(A, B, axis);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_flip_random(Tensor *A, Tensor *B, int axis){
  _profile_fpga(_FPGA_FLIP_RANDOM, 0);
#ifndef K_ENABLED_FLIP_RANDOM
  fpga_cpuemu_flip_random(A, B, axis);
#else
  printf("fpga_flip_random is not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_FLIP_RANDOM, 1);
}

// -----------------------------------------------------------------
// crop_random
//
void fpga_cpuemu_crop_random(Tensor *A, Tensor *B) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_crop_random(A, B);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_crop_random(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_CROP_RANDOM, 0);
#ifndef K_ENABLED_CROP_RANDOM
  fpga_cpuemu_crop_random(A, B);
#else
  printf("fpga_crop_random is not implemented yet\n"); exit(1);
#endif
  _profile_fpga(_FPGA_CROP_RANDOM, 1);
}

// -----------------------------------------------------------------
// crop_scale_random
//
void fpga_cpuemu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant) {
  fpga_copy_from_fpga(A, A->ptr);
  cpu_crop_scale_random(A, B, factor, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
  _profile_fpga(_FPGA_CROP_SCALE_RANDOM, 0);
#ifndef K_ENABLED_CROP_SCALE_RANDOM
  fpga_cpuemu_crop_scale_random(A, B, factor, mode, constant);
#else
  cl_int err;
  cl::Event event;

  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(0, *(A->fpga_ptr)));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(1, *(B->fpga_ptr)));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(2, A->shape[0]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(3, A->shape[2]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(4, A->shape[3]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(5, B->shape[0]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(6, B->shape[1]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(7, B->shape[2]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(8, B->shape[3]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(9, factor[0]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(10, factor[1]));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(11, A->stride[0]));    
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(12, A->stride[1])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(13, A->stride[2])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(14, A->stride[3])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(15, B->stride[0])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(16, B->stride[1])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(17, B->stride[2])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(18, B->stride[3])); 
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(19, mode));
  OCL_CHECK(err, err = kernel_crop_scale_random.setArg(20, constant));

  OCL_CHECK(err, err = q.enqueueTask(kernel_crop_scale_random, NULL, &event));
  q.finish();
#endif
  _profile_fpga(_FPGA_CROP_SCALE_RANDOM, 1);
}

// -----------------------------------------------------------------
// cutout_random
//
void fpga_cpuemu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
  fpga_copy_from_fpga(A, A->ptr);
  cpu_cutout_random(A, B, factor_x, factor_y, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
  _profile_fpga(_FPGA_CUTOUT_RANDOM, 0);
#ifndef K_ENABLED_CUTOUT_RANDOM
  fpga_cpuemu_cutout_random(A, B, factor_x, factor_y, constant);
#else
  printf("fpga_cutout_random is not implemented yet\n"); exit(1);
#endif
 _profile_fpga(_FPGA_CUTOUT_RANDOM, 0);
}

#endif
