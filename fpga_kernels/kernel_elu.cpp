#include <math.h>
#include <stdio.h>
extern "C" {

void k_elu(float *A, float *B, long int size, float param){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = param * (expf(A[i]) - 1.0);  // check expf is ok
  }
}

} // end extern "C"
