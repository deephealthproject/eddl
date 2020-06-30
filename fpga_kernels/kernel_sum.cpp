#include <math.h>
#include <stdio.h>
extern "C" {

void k_sum(float *A, long int size, float *sum) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=sum bundle=control

  float local_sum = 0.0;
  for (int i = 0; i < size; ++i) local_sum += A[i];
  *sum = local_sum;
  }

}
