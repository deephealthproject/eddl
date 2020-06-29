#include <math.h>
#include <stdio.h>
extern "C" {

#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_add(float scA, float *A, float scB, float *B, float *C, int incC, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=scA bundle=control
  #pragma HLS INTERFACE s_axilite port=scB bundle=control
  #pragma HLS INTERFACE s_axilite port=incC bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  float buffer_a[BUFFER_SIZE];
  float buffer_b[BUFFER_SIZE];
  float buffer_c[BUFFER_SIZE];
  float buffer_out[BUFFER_SIZE];

  for (int i=0; i<size; i=i+BUFFER_SIZE) {

    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;

    // burst read of data vectors from global memory
    read1:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      buffer_a[j] = A[i + j];
    }
    read2:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      buffer_b[j] = B[i + j];
    }
    read3:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      buffer_c[j] = C[i + j];
    }

    /* from xilinx forum
     * https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/Impact-of-if-else-in-for-loop/td-p/813177
     * Sometimes HLS will get confused because it's not sure whether it needs to readd D[] or not
     * because it doesn't need to read it if I[i]<=0
     * We write the loop so that it always reads D[i] - but ignores the result if I[i]<=0
     * It is counter-intuitive, but sometimes this is necessary for HLW
     */
    // prepare data
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      if (incC) buffer_c[j] = 0;
    }

    /*for (int i = 0; i < size; i++)
      if (incC) C[i] += scA * A[i] + scB * B[i];
      else C[i] = scA * A[i] + scB * B[i];*/
    add:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL FACTOR=2
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      // perform operation
      // NO support for native_fabs 
      buffer_out[j] = buffer_c[j] + (scA * buffer_a[j]) + (scB * buffer_b[j]);
    }

    // burst write the result
    write:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      C[i+j] = buffer_out[j];
    }
  }
} // end kernel function

} // end extern "C"
