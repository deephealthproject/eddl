#include <math.h>
#include <stdio.h>
extern "C"{

//from src/hardware/cpu/cpu_core.cpp
/*void cpu_fill(Tensor * A, int aini, int aend, Tensor * B, int bini, int bend, int inc){
    _profile(_CPU_FILL, 0);
    int at = A->size / A->shape[0];
    int bt = B->size / B->shape[0];

    int t = 1;
    for (int i = 2; i < A->ndim; i++)
        t *= A->shape[i];

#pragma omp parallel for
    for (int i = 0; i < A->shape[0]; i++) {
        int ap = (i * at) + (aini * t);
        int bp = (i * bt) + (bini * t);

        for (int j = aini; j < aend; j++) {
            for (int k = 0; k < t; k++, ap++, bp++)
                if (inc) B->ptr[bp] += A->ptr[ap];
                else B->ptr[bp] = A->ptr[ap];
        }
    }
    _profile(_CPU_FILL, 1);
}
*/
#define DATA_SIZE 4096
#define BUFFER_SIZE 1024

// TRIPCOUNT identifiers
const unsigned int c_chunk_sz = BUFFER_SIZE;
const unsigned int c_size = DATA_SIZE;

void k_fill(float *A, int aini, int aend, float *B, int bini, int bend, int inc, int ndim, int a_size, int a_shape_0, int b_size, int b_shape_0, int t) {
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=aini  bundle=control
  #pragma HLS INTERFACE s_axilite port=aend  bundle=control
  #pragma HLS INTERFACE s_axilite port=bini  bundle=control
  #pragma HLS INTERFACE s_axilite port=bend  bundle=control
  #pragma HLS INTERFACE s_axilite port=inc   bundle=control
  #pragma HLS INTERFACE s_axilite port=ndim  bundle=control
  #pragma HLS INTERFACE s_axilite port=a_size     bundle=control
  #pragma HLS INTERFACE s_axilite port=a_shape_0  bundle=control
  #pragma HLS INTERFACE s_axilite port=b_size     bundle=control
  #pragma HLS INTERFACE s_axilite port=b_shape_0  bundle=control
  #pragma HLS INTERFACE s_axilite port=t      bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  float buffer_a[BUFFER_SIZE];
  float buffer_b[BUFFER_SIZE];

  int at = a_size / a_shape_0;
  int bt = b_size / b_shape_0;

  for (int i=0; i<a_shape_0; i++) {
    #pragma HLS LOOP_TRIPCOUNT min=c_size/c_chunk_sz max=c_size/c_chunk_sz
    int ap = (i * at) + (aini * t);
    int bp = (i * bt) + (bini * t);

    //for (int j=aini; j<aend; j++) {
    for (int j=aini; j<aend; j=j+BUFFER_SIZE, ap=ap+BUFFER_SIZE, bp=bp+BUFFER_SIZE)
      // boundary checks
      int chunk_size = BUFFER_SIZE;
      //if ((i + BUFFER_SIZE) > size)
      if( (j+BUFFER_SIZE)> aend)
        chunk_size = aend - j;

      // burst read of data vectors from global memory
      read1:
      for (int k=0; k<chunk_size; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
        // set offset for A tensor
        buffer_a[k] = A[ap + k];
      }
      read2:
      for (int k=0; k<chunk_size; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
        // set offset for B tensor
        buffer_b[k] = B[bp + k];
      }

      k_fill:
      for (int k=0; k<t; k++) {
        if (inc) buffer_b[k] += buffer_a[k];
        else     buffer_b[k]  = buffer_a[k];
      }
      // burst write the result
      write:
      for (int k=0; k<chunk_size; k++) {
        #pragma HLS LOOP_TRIPCOUNT min=c_chunk_sz max=c_chunk_sz
        B[bp+k] = buffer_b[k];
      }     
    }
  }
} // end kernel

} // end extern "C"
