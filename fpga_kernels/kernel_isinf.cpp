#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

void k_isinf(float *A, float* B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
}
