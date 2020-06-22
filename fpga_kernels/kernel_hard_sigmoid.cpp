#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_HARD_SIGMOID
void k_hard_sigmoid(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > 2.5) B[i] = 1.0;
    else if (A[i] < -2.5) B[i] = 0.0;
    else B[i] = (0.2 * A[i]) + 0.5;
  }
}
#endif

}
