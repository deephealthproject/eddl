#include <math.h>
#include <stdio.h>
extern "C" {

void k_single_shift(int b, float *A, float *B, int *shift, int mode, float constant){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=shift offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=shift  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=mode bundle=control
  #pragma HLS INTERFACE s_axilite port=constant bundle=control

}

}
