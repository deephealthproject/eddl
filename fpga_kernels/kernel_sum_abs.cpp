#include <math.h>
#include <stdio.h>
extern "C" {

float k_sum_abs(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=sum bundle=control //return

  float sum = 0.0;
  for (int i = 0; i < size; ++i) sum += ::fabs(A[i]);
}

}
