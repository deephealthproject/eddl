#include <math.h>
#include <stdio.h>
extern "C" {

void k_pow_(float *A, float exp, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=exp bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
  // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
  // speed: 0.057887s
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = ::powf(A[i], exp);
}

}
