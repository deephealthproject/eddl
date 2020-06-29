#include <math.h>
#include <stdio.h>
extern "C"{

// from src/hardware/cpu/cpu_core.cpp
/*void cpu_fill_(Tensor *A, float v){
    _profile(_CPU_FILL_, 0);
    #pragma omp parallel for
    for (int i = 0; i < A->size; ++i){
        A->ptr[i] = v;
    }
    _profile(_CPU_FILL_, 1);
}
*/

// from src/harcdware/fpga/fpga_core.cpp
/*void fpga_fill_(Tensor *A, float v){
    OCL_CHECK(err, err = kernel_fill_.setArg(0, *(A->fpga_ptr)));
    OCL_CHECK(err, err = kernel_fill_.setArg(1, v));
    OCL_CHECK(err, err = kernel_fill_.setArg(2, (long int)A->size));
*/
#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_fill_(float *A, float v, long int size){
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control 

  float buffer[BUFFER_SIZE];

  for (int i=0; i<size; i=i+BUFFER_SIZE) {

    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
    int chunk_size = BUFFER_SIZE;
    // boundary checks
    if ((i + BUFFER_SIZE) > size)
      chunk_size = size - i;

    fill_:
    for (int j=0; j<chunk_size; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL FACTOR=2
      #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
      // perform operation
      buffer[j] = v;
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
