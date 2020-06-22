#include <math.h>
#include <stdio.h>
extern "C" {

void k_single_scale(int b, int *offsets, float *A, float *B, int *new_shape, int mode, float constant){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=offsets offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=new_shape offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=offsets  bundle=control
  #pragma HLS INTERFACE s_axilite port=new_shape  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=mode bundle=control
  #pragma HLS INTERFACE s_axilite port=constant bundle=control

}

}
