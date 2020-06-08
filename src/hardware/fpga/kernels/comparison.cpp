#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_ALL
bool k_all(float *A, long int size ){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ANY
bool k_any(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ISFINITE
void k_isfinite(float *A, float* B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ISINF
void k_isinf(float *A, float* B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ISNAN
void k_isnan(float *A, float* B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ISNEGINF
void k_isneginf(float *A, float* B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ISPOSINF
void k_isposinf(float *A, float* B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_LOGICAL_AND
void k_logical_and(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_LOGICAL_OR
void k_logical_or(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_LOGICAL_NOT
void k_logical_not(float *A, float *B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_LOGICAL_XOR
void k_logical_xor(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_ALLCLOSE
bool k_allclose(float *A, float *B, float rtol, float atol, bool equal_nan, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=rtol  bundle=control
  #pragma HLS INTERFACE s_axilite port=atol  bundle=control
  #pragma HLS INTERFACE s_axilite port=equal_nan  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
}
#endif

#ifdef K_ENABLED_ISCLOSE
void k_isclose(float *A, float *B, float *C, float rtol, float atol, bool equal_nan, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=rtol  bundle=control
  #pragma HLS INTERFACE s_axilite port=atol  bundle=control
  #pragma HLS INTERFACE s_axilite port=equal_nan  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_GREATER
void k_greater(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_GREATER_EQUAL
void k_greater_equal(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_LESS
void k_less(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_LESS_EQUAL
void k_less_equal(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_EQUAL
void k_equal(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_NOT_EQUAL
void k_not_equal(float *A, float *B, float *C, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

#ifdef K_ENABLED_EQUAL2
int k_equal2(float *A, float *B, float epsilon, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=epsilon  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

}
#endif

}
