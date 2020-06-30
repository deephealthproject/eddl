#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_REPEAT_NN
void k_repeat_nn(float *A, float *B, int *size_ptr){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=size_ptr offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size_ptr  bundle=control

}
#endif

#ifdef K_ENABLED_D_REPEAT_NN
void k_d_repeat_nn(float *D, float *A, int *size_ptr){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=size_ptr offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size_ptr  bundle=control

}
#endif

}
