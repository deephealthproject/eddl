#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_CENT
void k_cent(float *A, float *B, float *C, long int size) {

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control

// In the CPU version a constant value has been added (0.00001)
// To check whether an else is of need, seems strage not having the
// else

for (int i=0; i < size; i++) {
  C[i]=0;
  if (A[i] != 0.0) C[i] -= A[i] * log( B[i] + 0.00001 );
  if (A[i] != 1.0) C[i] -= (1.0 - A[i]) * log( 1.0 - B[i] + 0.00001 );
}
}
#endif

}
