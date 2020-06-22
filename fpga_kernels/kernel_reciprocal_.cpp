#include <math.h>
#include <stdio.h>
extern "C" {

void k_reciprocal_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = 1.0f/A[i];
}

}
