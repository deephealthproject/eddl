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

}
