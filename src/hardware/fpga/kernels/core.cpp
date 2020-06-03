#include <math.h>
#include <stdio.h>
extern "C" {

void k_fill_(float *A, float v){
}

void k_fill(float * A, int aini, int aend, float * B, int bini, int bend, int inc){
}

void k_select(float *A, float *B, void *sd){
}

void k_select_back(float *A, float *B, void *sd){
}

void k_set_select(float *A, float *B, void *sd){
}

void k_set_select_back(float *A, float *B, void *sd){
}

void k_select(float * A, float * B, int *sind, int ini, int end,bool mask_zeros){
}

void k_deselect(float * A, float * B, int *sind, int ini, int end,int inc,bool mask_zeros){
}

void k_concat(float *A, float **t, unsigned int axis, bool derivative){
}

}