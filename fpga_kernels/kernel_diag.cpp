#include <math.h>
#include <stdio.h>
extern "C" {

#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_diag(float *A, float *B, long int size, int Ashape0, int Ashape1, int k){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size    bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=k       bundle=control
  #pragma HLS INTERFACE s_axilite port=return  bundle=control 

  float buffer_a[BUFFER_SIZE];
  float buffer_b[BUFFER_SIZE];

  for (int i=0; i<size; i=i+BUFFER_SIZE) {

    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;

    // burst read of A vector from global memory
    read1:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      buffer_a[j] = A[i + j];
    }

    /*for(int i=0; i<A->size; i++){
        if ((i/A->shape[0]+k) == i%A->shape[1]){ B->ptr[i] = A->ptr[i]; }  // rows+offset == col?
        else { B->ptr[i] = 0.0f; }
    }*/
    diag:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL FACTOR=2
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
   
      int ind = i+j;
      #ifdef HLS_NATIVE_FUNCTION_ENABLE
      int auxval = native_divide(ind, Ashape0);
      #else
      int auxval = ind/Ashape0;
      #endif
      // perform operation
      if ((auxval + k) == fmod(ind,Ashape1)) buffer_b[j] = buffer_a[j];
      else  buffer_b[j] = 0.0f;
    }

    // burst write the result
    write:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      B[i+j] = buffer[j];
    }
  }
} // end kernel

} // end extern "C"
