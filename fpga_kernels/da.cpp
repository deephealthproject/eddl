#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_SINGLE_SHIFT
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
#endif

#ifdef K_ENABLED_SINGLE_ROTATE
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
#endif

#ifdef K_ENABLED_SINGLE_SCALE
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
#endif

#ifdef K_ENABLED_SINGLE_FLIP
void k_single_flip(int b, bool apply, float *A, float *B, int axis){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=apply bundle=control
  #pragma HLS INTERFACE s_axilite port=axis bundle=control

}
#endif

#ifdef K_ENABLED_SINGLE_CROP
void k_single_crop(int b, const int *offsets, float *A, float *B, int *coords_from, int *coords_to, float constant, bool inverse){

  #pragma HLS INTERFACE m_axi port=offsets offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=coords_from offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=coords_to offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=offsets  bundle=control
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=coords_from  bundle=control
  #pragma HLS INTERFACE s_axilite port=coords_to  bundle=control
  #pragma HLS INTERFACE s_axilite port=b bundle=control
  #pragma HLS INTERFACE s_axilite port=constant bundle=control
  #pragma HLS INTERFACE s_axilite port=inverse bundle=control

}
#endif

#ifdef K_ENABLED_SINGLE_CROP_SCALE
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
#endif

}
