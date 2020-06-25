#include <math.h>
#include <stdio.h>
extern "C" {

void k_clamp_(float *A, float min, float max, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control
  #pragma HLS INTERFACE s_axilite port=max bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i){
    if (A[i] < min){
      A[i] = min;
    } else if(A[i] > max){
      A[i] = max;
    }
  }
}

}
