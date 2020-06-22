#include <math.h>
#include <stdio.h>
extern "C" {

void k_sum2D_colwise(float *A, float *B, float *C, int Ashape0, int Ashape1) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control

  for (int i = 0; i < Ashape0; i++) {
    int p=i*Ashape1;
    for (int j = 0; j < Ashape1; j++, p++)
    C[p] = A[p] + B[i];
  }
}

}
