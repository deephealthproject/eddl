#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

void k_isclose(float *A, float *B, float *C, float rtol, float atol, bool equal_nan, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=rtol  bundle=control
  #pragma HLS INTERFACE s_axilite port=atol  bundle=control
  #pragma HLS INTERFACE s_axilite port=equal_nan  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
}
