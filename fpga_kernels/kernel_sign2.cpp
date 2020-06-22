#include <math.h>
#include <stdio.h>
extern "C" {

void k_sign2(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
  if (A[i] < 0) B[i] = -1.0;
  else B[i] = 1.0;
}

}
