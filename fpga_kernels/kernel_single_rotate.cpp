#include <math.h>
#include <stdio.h>
extern "C" {

void k_single_rotate(int b, float *A, float *B, float angle, int *offset_center, int mode, float constant){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=offset_center offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=offset_center  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=angle bundle=control
  #pragma HLS INTERFACE s_axilite port=mode bundle=control
  #pragma HLS INTERFACE s_axilite port=constant bundle=control

}

}
