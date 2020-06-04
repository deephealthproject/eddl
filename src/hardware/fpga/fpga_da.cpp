/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/




#include <iostream>
#include <utility>
#include <cmath>

#include "eddl/hardware/fpga/fpga_hw.h"
#include "eddl/random.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// emulation switches of functions (via cpu)
// when set the function is run on the cpu
char fpga_set_cpuemu_single_shift       = 1;
char fpga_set_cpuemu_single_rotate      = 1;
char fpga_set_cpuemu_single_scale       = 1;
char fpga_set_cpuemu_single_flip        = 1;
char fpga_set_cpuemu_single_crop        = 1;
char fpga_set_cpuemu_single_crop_scale  = 1;
char fpga_set_cpuemu_shift              = 1;
char fpga_set_cpuemu_rotate             = 1;
char fpga_set_cpuemu_scale              = 1;
char fpga_set_cpuemu_flip               = 1;
char fpga_set_cpuemu_crop               = 1;
char fpga_set_cpuemu_crop_scale         = 1;
char fpga_set_cpuemu_shift_random       = 1;
char fpga_set_cpuemu_flip_random        = 1;
char fpga_set_cpuemu_crop_random        = 1;
char fpga_set_cpuemu_rotate_random      = 1;
char fpga_set_cpuemu_scale_random       = 1;
char fpga_set_cpuemu_crop_scale_random  = 1;
char fpga_set_cpuemu_cutout_random      = 1;

// -----------------------------------------------------------------
// single_shift
//
void fpga_cpuemu_single_shift(int b, Tensor *A, Tensor *B, vector<int> shift, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_single_shift(b, A, B, shift, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_single_shift(int b, Tensor *A, Tensor *B, vector<int> shift, int mode, float constant){
    _profile_fpga(_FPGA_SINGLE_SHIFT, 0);
    if (fpga_set_cpuemu_single_shift == 1) {
        fpga_cpuemu_single_shift(b, A, B, shift, mode, constant);
    } else {
        printf("fpga_single_shift not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINGLE_SHIFT, 1);
}

// -----------------------------------------------------------------
// single_rotate
//
void fpga_cpuemu_single_rotate(int b, Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_single_rotate(b, A, B, angle, offset_center, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_single_rotate(int b, Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant){
    _profile_fpga(_FPGA_SINGLE_ROTATE, 0);
    if (fpga_set_cpuemu_single_rotate == 1) {
        fpga_cpuemu_single_rotate(b, A, B, angle, offset_center, mode, constant);
    } else {
        printf("fpga_single_rotate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINGLE_ROTATE, 1);
}

// -----------------------------------------------------------------
// single_scale
//
void fpga_cpuemu_single_scale(int b, int* offsets, Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_single_scale(b, offsets, A, B, new_shape, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_single_scale(int b, int* offsets, Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant){
    _profile_fpga(_FPGA_SINGLE_SCALE, 0);
    if (fpga_set_cpuemu_single_scale == 1) {
        fpga_cpuemu_single_scale(b, offsets, A, B, new_shape, mode, constant);
    } else {
        printf("fpga_single_scale not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINGLE_SCALE, 1);
}

// -----------------------------------------------------------------
// single_flip
//
void fpga_cpuemu_single_flip(int b, bool apply, Tensor *A, Tensor *B, int axis){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_single_flip(b, apply, A, B, axis);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_single_flip(int b, bool apply, Tensor *A, Tensor *B, int axis){
    _profile_fpga(_FPGA_SINGLE_FLIP, 0);
    if (fpga_set_cpuemu_single_flip == 1) {
        fpga_cpuemu_single_flip(b, apply, A, B, axis);
    } else {
        printf("fpga_single_flip not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINGLE_FLIP, 1);
}

// -----------------------------------------------------------------
// single_crop
//
void fpga_cpuemu_single_crop(int b, const int* offsets, Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_single_crop(b, offsets, A, B, coords_from, coords_to, constant, inverse);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_single_crop(int b, const int* offsets, Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
    _profile_fpga(_FPGA_SINGLE_CROP, 0);
    if (fpga_set_cpuemu_single_crop == 1) {
        fpga_cpuemu_single_crop(b, offsets, A, B, coords_from, coords_to, constant, inverse);
    } else {
        printf("fpga_single_crop not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINGLE_CROP, 1);
}

// -----------------------------------------------------------------
// single_crop_scale
//
void fpga_cpuemu_single_crop_scale(int b, Tensor* A, Tensor* B, vector<int> coords_from, vector<int> coords_to, int mode, float constant){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_single_crop_scale(b, A, B, coords_from, coords_to, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_single_crop_scale(int b, Tensor* A, Tensor* B, vector<int> coords_from, vector<int> coords_to, int mode, float constant){
    _profile_fpga(_FPGA_SINGLE_CROP_SCALE, 0);
    if (fpga_set_cpuemu_single_crop_scale == 1) {
        fpga_cpuemu_single_crop_scale(b, A, B, coords_from, coords_to, mode, constant);
    } else {
        printf("fpga_single_crop_scale not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SINGLE_CROP_SCALE, 1);
}

// CPU: Data augmentation (2D Optimized) ********************************************

// -----------------------------------------------------------------
// single_shift
//
void fpga_cpuemu_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_shift(A, B, shift, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_shift(Tensor *A, Tensor *B, vector<int> shift, int mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html
    _profile_fpga(_FPGA_SHIFT, 0);
    if (fpga_set_cpuemu_shift == 1) {
        fpga_cpuemu_shift(A, B, shift, mode, constant);
    } else {
        printf("fpga_shift ate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SHIFT, 1);
}

// -----------------------------------------------------------------
// rotate
//
void fpga_cpuemu_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_rotate(A, B, anglre, offset_center, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_rotate(Tensor *A, Tensor *B, float angle, vector<int> offset_center, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    _profile_fpga(_FPGA_ROTATE, 0);
    if (fpga_set_cpuemu_rotate == 1) {
        fpga_cpuemu_rotate(A, B, angle, offset_center, mode, constant);
    } else {
        printf("fpga_rotate ate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ROTATE, 1);
}

// -----------------------------------------------------------------
// scale
//
void fpga_cpuemu_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_scale(A, B, new_shape, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_scale(Tensor *A, Tensor *B, vector<int> new_shape, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    // I use "new_shape" because I might want to keep the shape of B, but thinking of it as a bigger/smaller matrix
    // If the new_shape is smaller than B, performs a downscale with padding
    // For cases:
    // A=5x5; B=10x10; new_size=10x10 => Normal zoom
    // A=5x5; B=5x5; new_size=5x5 => Normal zoom-out
    // A=10x10; B=10x10; new_size=5x5 => Zoom-out centered
    // A=5x5; B=5x5; new_size=10x10 => Zoom in window

    _profile_fpga(_FPGA_SCALE, 0);
    if (fpga_set_cpuemu_scale == 1) {
        fpga_cpuemu_scale(A, B, new_shape, mode, constant);
    } else {
        printf("fpga_scale ate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SCALE, 1);
}

// -----------------------------------------------------------------
// flip
//
void fpga_cpuemu_flip(Tensor *A, Tensor *B, int axis){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_flip(A, B, axis);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_flip(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    _profile_fpga(_FPGA_FLIP, 0);
    if (fpga_set_cpuemu_flip == 1) {
        fpga_cpuemu_flip(A, B, axis);
    } else {
        printf("fpga_flip ate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_FLIP, 1);
}

// -----------------------------------------------------------------
// crop
//
void fpga_cpuemu_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_crop(A, B, coords_from, coords_to, constant, inverse);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_crop(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, float constant, bool inverse){
    // Two cases:
    // -> A=10x10; B=3x3; crop_size=3x3 => Normal crop
    // -> A=10x10; B=10x10; crop_size=3x3 => Crop with padding (inverse of cutout)
    // Inverse => For cutout

    _profile_fpga(_FPGA_CROP, 0);
    if (fpga_set_cpuemu_crop == 1) {
        fpga_cpuemu_crop(A, B, coords_from, coords_to, constant, inverse);
    } else {
        printf("fpga_crop ate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CROP, 1);
}

// -----------------------------------------------------------------
// crop-scale
//
void fpga_cpuemu_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_crop_scale(A, B, coords_from, coords_to, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_crop_scale(Tensor *A, Tensor *B, vector<int> coords_from, vector<int> coords_to, int mode, float constant){
    _profile_fpga(_FPGA_CROP_SCALE, 0);
    if (fpga_set_cpuemu_crop_scale == 1) {
        fpga_cpuemu_crop_scale(A, B, coords_from, coords_to, mode, constant);
    } else {
        printf("fpga_crop_scale ate not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CROP_SCALE, 1);
}


// FPGA: Data augmentation (2D Optimized) ********************************************

// -----------------------------------------------------------------
// shift_random
//
void fpga_cpuemu_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_shift_random(A, B, factor_x, factor_y, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_shift_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, int mode, float constant) {
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.shift.html

    _profile_fpga(_FPGA_SHIFT_RANDOM, 0);
    if (fpga_set_cpuemu_shift_random == 1) {
        fpga_cpuemu_shift_random(A, B, factor_x, factor_y, mode, constant);
    } else {
        printf("fpga_shift_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SHIFT_RANDOM, 1);
}

// -----------------------------------------------------------------
// rotate_random
//
void fpga_cpuemu_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant) {
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_rotate_random(A, B, factor, offset_center, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_rotate_random(Tensor *A, Tensor *B, vector<float> factor, vector<int> offset_center, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html
    _profile_fpga(_FPGA_ROTATE_RANDOM, 0);
    if (fpga_set_cpuemu_rotate_random == 1) {
        fpga_cpuemu_rotate_random(A, B, factor, offset_center, mode, constant);
    } else {
        printf("fpga_rotate_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_ROTATE_RANDOM, 1);
}


// -----------------------------------------------------------------
// scale_random
//
void fpga_cpuemu_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
  int Asize = A->size * sizeof(float);
  int Bsize = B->size * sizeof(float);
  if (A->ptr == NULL) A->ptr = (float *)malloc(Asize);
  if (B->ptr == NULL) B->ptr = (float *)malloc(Bsize);
  fpga_copy_from_fpga(A, A->ptr);
  cpu_scale_random(A, B, factor, mode, constant);
  fpga_copy_to_fpga(B->ptr, B);
}

void fpga_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
    // https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
    // I use "new_shape" because I might want to keep the shape of B, but thinking of it as a bigger/smaller matrix
    // If the factor is less than 1.0f, performs a downscale with padding

    _profile_fpga(_FPGA_SCALE_RANDOM, 0);
    if (fpga_set_cpuemu_scale_random == 1) {
        fpga_cpuemu_scale_random(A, B, factor, mode, constant);
    } else {
        printf("fpga_scale_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_SCALE_RANDOM, 1);
}

// -----------------------------------------------------------------
// flip_random
//
void fpga_cpuemu_flip_random(Tensor *A, Tensor *B, int axis) {
    printf("fpga_cpuemu_flip_random not implemented yet\n");
    exit(1);
}

void fpga_flip_random(Tensor *A, Tensor *B, int axis){
    // https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html


    _profile_fpga(_FPGA_FLIP_RANDOM, 0);
    if (fpga_set_cpuemu_flip_random == 1) {
        fpga_cpuemu_flip_random(A, B, axis);
    } else {
        printf("fpga_flip_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_FLIP_RANDOM, 1);
}

// -----------------------------------------------------------------
// crop_random
//
void fpga_cpuemu_crop_random(Tensor *A, Tensor *B) {
    printf("fpga_cpuemu_crop_random not implemented yet\n");
    exit(1);
}

void fpga_crop_random(Tensor *A, Tensor *B){
    // Performs a crop with padding (Keeps the original size)

    _profile_fpga(_FPGA_CROP_RANDOM, 0);
    if (fpga_set_cpuemu_crop_random == 1) {
        fpga_cpuemu_crop_random(A, B);
    } else {
        printf("fpga_crop_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CROP_RANDOM, 1);
}

// -----------------------------------------------------------------
// crop_scale_random
//
void fpga_cpuemu_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant) {
    printf("fpga_cpuemu_crop_scale_random not implemented yet\n");
    exit(1);
}

void fpga_crop_scale_random(Tensor *A, Tensor *B, vector<float> factor, int mode, float constant){
    _profile_fpga(_FPGA_CROP_SCALE_RANDOM, 0);
    if (fpga_set_cpuemu_crop_scale_random == 1) {
        fpga_cpuemu_crop_scale_random(A, B, factor, mode, constant);
    } else {
        printf("fpga_crop_scale_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CROP_SCALE_RANDOM, 1);
}

// -----------------------------------------------------------------
// cutout_ranom
//
void fpga_cpuemu_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    printf("fpga_cpuemu_cutout_random not implemented yet\n");
    exit(1);
}

void fpga_cutout_random(Tensor *A, Tensor *B, vector<float> factor_x, vector<float> factor_y, float constant){
    // Performs a crop with padding (Keeps the original size)

    _profile_fpga(_FPGA_CUTOUT_RANDOM, 0);
    if (fpga_set_cpuemu_cutout_random == 1) {
        fpga_cpuemu_cutout_random(A, B, factor_x, factor_y, constant);
    } else {
        printf("fpga_cutout_random is not implemented yet\n"); exit(1);
    }
    _profile_fpga(_FPGA_CUTOUT_RANDOM, 0);
}