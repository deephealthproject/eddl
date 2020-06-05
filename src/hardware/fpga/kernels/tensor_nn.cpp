#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_REPEAT_NN
void k_repeat_nn(float *A, float *B, int *size_ptr){
}
#endif

#ifdef K_ENABLED_D_REPEAT_NN
void k_d_repeat_nn(float *D, float *A, int *size_ptr){
}
#endif

}
