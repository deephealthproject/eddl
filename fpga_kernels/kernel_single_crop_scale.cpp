#include <math.h>
#include <stdio.h>
extern "C" {

void k_single_crop_scale(int b, float *A, float* B, int *coords_from, int *coords_to, int mode, float constant){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=coords_from offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=coords_to offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=coords_from  bundle=control
  #pragma HLS INTERFACE s_axilite port=coords_to  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=constant bundle=control
  #pragma HLS INTERFACE s_axilite port=mode bundle=control

}

}
