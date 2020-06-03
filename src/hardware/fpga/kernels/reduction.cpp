#include <math.h>
#include <stdio.h>
extern "C" {


void fpga_reduce(float *A, float *B, int mode, int* map) {
}

void fpga_reduce2(float *A, float *B, int mode, void *MD) {
}

void fpga_reduce_op(float *A, float *B, int op, int* map) {
}

void fpga_reduce_op2(float *A, float *B, int op, void *MD) {
}

void fpga_reduce_sum2D(float *A, float *B, int axis, int incB) {
}

void fpga_reduction(void *RD) {
}

void fpga_reduction_back(void *RD) {
}
