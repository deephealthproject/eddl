#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_REDUCE
void fpga_reduce(float *A, float *B, int mode, int* map) {
}
#endif

#ifdef K_ENABLED_REDUCE2
void fpga_reduce2(float *A, float *B, int mode, void *MD) {
}
#endif

#ifdef K_ENABLED_REDUCE_OP
void fpga_reduce_op(float *A, float *B, int op, int* map) {
}
#endif

#ifdef K_ENABLED_OPT2
void fpga_reduce_op2(float *A, float *B, int op, void *MD) {
}
#endif

#ifdef K_ENABLED_REDUCE_SUM2D
void fpga_reduce_sum2D(float *A, float *B, int axis, int incB) {
}
#endif

#ifdef K_ENABLED_REDUCTION
void fpga_reduction(void *RD) {
}
#endif

#ifdef K_ENABLED_REDUCTION_BACK
void fpga_reduction_back(void *RD) {
}
#endif

}
