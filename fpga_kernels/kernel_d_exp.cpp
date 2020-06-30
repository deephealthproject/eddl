#include <math.h>
#include <stdio.h>
extern "C" {

/*void k_d_exp(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  for (int i = 0; i < size; i++)
    PD[i] += D[i] * I[i];
}*/

#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_d_exp(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  float buffer_d[BUFFER_SIZE];
  float buffer_i[BUFFER_SIZE];
  float buffer_pd[BUFFER_SIZE];

  for(int i=0; i<size; i=i+BUFFER_SIZE) {
    
    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/chunck_sz
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;

    // burst read of vectors from global memory
    read1:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      buffer_d[j] = D[i + j];
    }
    read2:
    for(int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_chunk_sz
      buffer_i[j] = I[i + j];
    }
    read3:
    for(int j=0; j<chunk_size; j++) {
      #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_chunk_sz
      buffer_pd[j] = PD[i + j];
    }

    /*for (int i = 0; i < size; i++)
    PD[i] += D[i] * I[i];*/
    d_exp:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL FACTOR=2
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      // perform operation
      buffer_pd[j] = buffer_pd[j] + (buffer_d[j] * buffer_i[j]);
    }

    // burst write the result
    write:
    for(int j=0; j<chunk_size; j++) {
      PD[i+j] = buffer_pd[j];
    }
  }
}

} // end extern "C"
