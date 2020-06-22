#include <math.h>
#include <stdio.h>
extern "C" {

void k_normalize_(float *A, float min, float max, long int size, float Amin, float Amax){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control
  #pragma HLS INTERFACE s_axilite port=max bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=Amin bundle=control
  #pragma HLS INTERFACE s_axilite port=Amax bundle=control

  // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
  // (max2-min2)/(max1-min1) * (x-min1) + min2
  float max_ori = Amax;
  float min_ori = Amin;
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = (max-min)/(max_ori-min_ori) * (A[i]-min_ori) + min;
}

}
