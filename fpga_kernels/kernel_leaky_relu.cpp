#include <math.h>
#include <stdio.h>
extern "C" {

/*void k_leaky_relu(float *A, float *B, long int size, float param){
  
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = param*A[i];
  }
}*/

#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_leaky_relu(float *A, float *B, long int size, float param){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A bundle=control
  #pragma HLS INTERFACE s_axilite port=B bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control 

  float buffer_a[BUFFER_SIZE];
  float buffer_b[BUFFER_SIZE];

  for (int i=0; i<size; i=i+BUFFER_SIZE) {
    
    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz 
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;
    
    // burst read of A vector from global memory
    readA:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      buffer_a[j] = A[i + j];
    }
    
    /*for (int i = 0; i < size; i++) {
      if (A[i] > 0.0) B[i] = A[i];
      else B[i] = param*A[i];
    }*/
    leaky_relu:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL FACTOR=2
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      // perform relu
      if (buffer_a[j] > 0.0) buffer_b[j] = buffer_a[j];
      else buffer_b[j] = param * buffer_a[j];
    }
    
    // burst write the result
    write:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      B[i+j] = buffer_b[j];
    }
  }
}

} // end extern "C"
