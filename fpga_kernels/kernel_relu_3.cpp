#include <math.h>
#include <stdio.h>
#include <ap_int.h>

extern "C" {

#define DATA_SIZE 4096
#define BUFFER_SIZE 1024
#define DATAWIDTH 512
#define DATATYPE_SIZE 32
#define VECTOR_SIZE  (DATAWIDTH / DATATYPE_SIZE) // vector size is 16 (512/32 = 16)

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;
const unsigned int unroll_factor = 2;
const unsigned int c_chunk_sz_unroll = BUFFER_SIZE / unroll_factor;

void k_relu(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  float buffer[BUFFER_SIZE];
  #pragma HLS ARRAY_PARTITION variable=buffer dim=0 cyclic factor = 8



  Chunck_loop: for (int i=0; i<size; i=i+BUFFER_SIZE) {
    #pragma HLS pipeline

    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;

    // burst read of A vector from global memory
    read:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      // #pragma HLS UNROLL FACTOR = unroll_factor
      buffer[j] = A[i + j];
    }

    // relu_ext:
    // for (int i = 0; i < unroll_factor; i++){
      // #pragma HLS LOOP_TRIPCOUNT min=unroll_factor max=unroll_factor
      // #pragma HLS UNROLL FACTOR = unroll_factor
      relu_int: for (int j=0; j< chunk_size /*c_chunk_sz_unroll*/; j++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL FACTOR = 2
        #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
        // perform operation
        if (buffer[j /*+ c_chunk_sz_unroll*i*/] < 0.0) buffer[j /*+ c_chunk_sz_unroll*i*/] = 0.0f;
      }
  }
}

} // end extern "C"
