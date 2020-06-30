#include <math.h>
#include <stdio.h>
extern "C" {

#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_eye(float *A, int offset, long int size, int Ashape0, int Ashape1){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=offset bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  float buffer[BUFFER_SIZE];
  
  for (int i=0; i<size; i=i+BUFFER_SIZE) {
    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;

    /*for(int i=0; i<size; i++){
      if ((i / Ashape0 + offset) == i % Ashape1) { A[i] = 1.0f; }  // rows+offset == col?
      else { A[i] = 0.0f; }
      }*/
    eye:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL FACTOR=2
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      // perform operation
      int ind = i+j;
      #ifdef HLS_NATIVE_FUNCTION_ENABLE
      int auxva1 = native_divide((ind),Ashape0);
      #else
      int auxval = (ind)/Ashape0;
      #endif
      if ((auxval + offset) == fmod((ind), Ashape1)) buffer[j]= 1.0;
      else buffer[j] = 0.0f;
      
    }

    // burst write the result
    write:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      A[i+j] = buffer[j];
    }
  }
} // end kernel

} // end extern "C"
