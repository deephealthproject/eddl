#include <math.h>
#include <stdio.h>
extern "C" {

void k_logn_(float *A, float n, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=n bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::logf(A[i])/::logf(n);
}

}
