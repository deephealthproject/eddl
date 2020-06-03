#include <math.h>
#include <stdio.h>
extern "C" {

void k_single_shift(int b, float *A, float *B, int *shift, int mode, float constant){
}

void k_single_rotate(int b, float *A, float *B, float angle, int *offset_center, int mode, float constant){
}

void k_single_scale(int b, int* offsets, float *A, float *B, int *new_shape, int mode, float constant){
}

void k_single_flip(int b, bool apply, float *A, float *B, int axis){
}

void k_single_crop(int b, const int* offsets, float *A, float *B, int *coords_from, int *coords_to, float constant, bool inverse){
}

void k_single_crop_scale(int b, float* A, float* B, int *coords_from, int *coords_to, int mode, float constant){
}

void k_shift(float *A, float *B, int *shift, int mode, float constant) {
}

void k_rotate(float *A, float *B, float angle, int *offset_center, int mode, float constant){
}

void k_scale(float *A, float *B, int *new_shape, int mode, float constant){
}

void k_flip(float *A, float *B, int axis){
}

void k_crop(float *A, float *B, int *coords_from, int *coords_to, float constant, bool inverse){
}

void k_crop_scale(float *A, float *B, int *coords_from, int *coords_to, int mode, float constant){
}

void k_shift_random(float *A, float *B, float *factor_x, float *factor_y, int mode, float constant) {
}

void k_rotate_random(float *A, float *B, float *factor, int *offset_center, int mode, float constant){
}

void k_scale_random(float *A, float *B, float *factor, int mode, float constant){
}

void k_flip_random(float *A, float *B, int axis){
}

void k_crop_random(float *A, float *B){
}

void k_crop_scale_random(float *A, float *B, float *factor, int mode, float constant){
}

void k_cutout_random(float *A, float *B, float *factor_x, float *factor_y, float constant){
}

}