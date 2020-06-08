#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_REDUCE
void fpga_reduce(float *A, float *B, int mode, int *map) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=map offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=map bundle=control
  #pragma HLS INTERFACE s_axilite port=mode bundle=control


}
#endif

#ifdef K_ENABLED_REDUCE2
void fpga_reduce2(float *A, float *B, int mode, void *MD) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=MD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=MD bundle=control
  #pragma HLS INTERFACE s_axilite port=mode bundle=control

}
#endif

#ifdef K_ENABLED_REDUCE_OP
void fpga_reduce_op(float *A, float *B, int op, int* map) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=map offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=map bundle=control
  #pragma HLS INTERFACE s_axilite port=op bundle=control

}
#endif

#ifdef K_ENABLED_OPT2
void fpga_reduce_op2(float *A, float *B, int op, void *MD) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=MD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=MD bundle=control
  #pragma HLS INTERFACE s_axilite port=op bundle=control
}
#endif

#ifdef K_ENABLED_REDUCE_SUM2D
void fpga_reduce_sum2D(float *A, float *B, int axis, int incB) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=axis bundle=control
  #pragma HLS INTERFACE s_axilite port=incB bundle=control

}
#endif

#ifdef K_ENABLED_REDUCTION
void fpga_reduction(void *RD) {

  #pragma HLS INTERFACE m_axi port=RD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=RD  bundle=control

}
#endif

#ifdef K_ENABLED_REDUCTION_BACK
void fpga_reduction_back(void *RD) {

  #pragma HLS INTERFACE m_axi port=RD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=RD  bundle=control
  
}
#endif

}
