#include <math.h>
#include <stdio.h>
extern "C" {

void k_eye(float *A, int offset, long int size, int Ashape0, int Ashape1){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=offset bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for(int i=0; i<size; i++){
        if ((i / Ashape0 + offset) == i % Ashape1) { A[i] = 1.0f; }  // rows+offset == col?
        else { A[i] = 0.0f; }
    }
}

}
