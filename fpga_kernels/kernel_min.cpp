#include <math.h>
#include <stdio.h>
extern "C" {

float k_min(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control //return

  float min = A[0];
  // todo: #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    if (A[i] < min) { min = A[i]; }
  }
  }

}
