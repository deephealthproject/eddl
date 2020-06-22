#include <math.h>
#include <stdio.h>
extern "C" {

void k_single_flip(int b, bool apply, float *A, float *B, int axis){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=apply bundle=control
  #pragma HLS INTERFACE s_axilite port=axis bundle=control

}

}
