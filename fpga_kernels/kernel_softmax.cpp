#include <math.h>
#include <stdio.h>
extern "C" {

// #define DATA_SIZE 4096
// #define BUFFER_SIZE 1024

// // TRIPCOUNT identifiers
// const unsigned int c_chunk_sz = BUFFER_SIZE;
// const unsigned int c_size = DATA_SIZE;

void k_softmax(float *A, float *B, int Ashape0, int Ashape1, int Bshape1) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  float max;
  float sum;
  // float buffer[BUFFER_SIZE];

  // for (int i=0; i<size; i=i+BUFFER_SIZE) {
  //
  //   #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
  //   int chunk_size = BUFFER_SIZE;
  //   // boundary checks
  //   if ((i + BUFFER_SIZE) > size)
  //     chunk_size = size - i;

    // burst read of A vector from global memory
    // read1:
    // for (int j=0; j<chunk_size; j++) {
    //   #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
    //   buffer[j] = A[i + j];
    // }
    //
    // printf("A\n");
    // imprimir:
    // for (int j = 0; j<chunk_size; j++){
    //   printf("%lf ", buffer[j]);
    // }

    softmax:
    for(int i = 0; i<Ashape0;i++){
      max = A[i];
      for(int j = 0; j < Ashape1; j++){
        if(A[i * Ashape1 +j] > max) {
          max = A[i * Ashape1 + j ];
        }
      }
      sum = 0;
      for(int j = 0; j < Ashape1; j++) {
        B[ i * Ashape1 + j ] = exp( A [ i * Ashape1 + j ] - max );
        sum = sum + B[ i * Ashape1 + j ];
      }
      for (int j=0; j < Ashape1; j++) {
        B[ i * Ashape1 + j ] = B[ i * Ashape1 + j] / sum;
      }
    }

    // burst write the result
    // write:
    // for (int j=0; j<chunk_size; j++) {
    //   #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
    //   B[i+j] = buffer[j];
    // }


  }
}
