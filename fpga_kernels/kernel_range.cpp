#include <math.h>
#include <stdio.h>
extern "C" {

void k_range(float *A, float min, float step, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control
  #pragma HLS INTERFACE s_axilite port=step bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

   float v=min;

    for(int i=0; i<size; i++){
        A[i] = v;
        v+=step;
    }
}

}
