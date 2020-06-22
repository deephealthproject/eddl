#include <math.h>
#include <stdio.h>
extern "C" {

void k_softmax(float *A, float *B, int Ashape0, int Ashape1, int Bshape1) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control

  #pragma HLS INLINE
  float max;
  float sum;

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

  // check this function as the cpu one uses ptr2 pointers and Bshape instead of Ashape!!!!
  // Below is the cpu code
  //for (int i = 0; i < A->shape[0]; i++) {
  //  max = (*A->ptr2).col(i).maxCoeff();
  //  for (int j = 0; j < A->shape[1]; j++)
  //  (*B->ptr2)(j, i) = std::exp((*A->ptr2)(j, i) - max);
  //
  //  sum = (*B->ptr2).col(i).sum();
  //  for (int j = 0; j < B->shape[1]; j++)
  //  (*B->ptr2)(j, i) = (*B->ptr2)(j, i) / sum;
  //}

  }
}
