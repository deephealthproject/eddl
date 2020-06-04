#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_SINGLE_SHIFT
void k_single_shift(int b, float *A, float *B, int *shift, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SINGLE_ROTATE
void k_single_rotate(int b, float *A, float *B, float angle, int *offset_center, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SINGLE_SCALE
void k_single_scale(int b, int* offsets, float *A, float *B, int *new_shape, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SINGLE_FLIP
void k_single_flip(int b, bool apply, float *A, float *B, int axis){
}
#endif

#ifdef K_ENABLED_SINGLE_CROP
void k_single_crop(int b, const int* offsets, float *A, float *B, int *coords_from, int *coords_to, float constant, bool inverse){
}
#endif

#ifdef K_ENABLED_SINGLE_CROP_SCALE
void k_single_crop_scale(int b, float* A, float* B, int *coords_from, int *coords_to, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SHIFT
void k_shift(float *A, float *B, int *shift, int mode, float constant) {
}
#endif

#ifdef K_ENABLED_ROTATE
void k_rotate(float *A, float *B, float angle, int *offset_center, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SCALE
void k_scale(float *A, float *B, int *new_shape, int mode, float constant){
}
#endif

#ifdef K_ENABLED_FLIP
void k_flip(float *A, float *B, int axis){
}
#endif

#ifdef K_ENABLED_CROP
void k_crop(float *A, float *B, int *coords_from, int *coords_to, float constant, bool inverse){
}
#endif

#ifdef K_ENABLED_CROP_SCALE
void k_crop_scale(float *A, float *B, int *coords_from, int *coords_to, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SHIFT_RANDOM
void k_shift_random(float *A, float *B, float *factor_x, float *factor_y, int mode, float constant) {
}
#endif

#ifdef K_ENABLED_ROTATE_RANDOM
void k_rotate_random(float *A, float *B, float *factor, int *offset_center, int mode, float constant){
}
#endif

#ifdef K_ENABLED_SCALE_RANDOM
void k_scale_random(float *A, float *B, float *factor, int mode, float constant){
}
#endif

#ifdef K_ENABLED_FLIP_RANDOM
void k_flip_random(float *A, float *B, int axis){
}
#endif

#ifdef K_ENABLED_CROP_RANDOM
void k_crop_random(float *A, float *B){
}
#endif

#ifdef K_ENABLED_CROP_SCALE_RANDOM
void k_crop_scale_random(float *A, float *B, float *factor, int mode, float constant){
}
#endif

#ifdef K_ENABLED_CUTOUT_RANDOM
void k_cutout_random(float *A, float *B, float *factor_x, float *factor_y, float constant){
}
#endif

}
