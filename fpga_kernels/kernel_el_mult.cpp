#include <math.h>
#include <stdio.h>
extern "C" {

void k_el_mult(float *A, float *B, float *C, int incC, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=incC bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
  if (incC) C[i] += A[i] * B[i];
  else C[i] = A[i] * B[i];
}

}
