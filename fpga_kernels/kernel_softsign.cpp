#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_SOFTSIGN
void k_softsign(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        B[i] = A[i] / (1 + fabs(A[i]));  // check fabs
    }
}
#endif

} // end extern "C"
