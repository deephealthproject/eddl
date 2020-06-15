#include <math.h>
#include <stdio.h>
#include "../../../../include/eddl/hardware/fpga/fpga_enables.h"
extern "C" {

#ifdef K_ENABLED_ABS_
void k_abs_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = fabs(A[i]);
}
#endif

#ifdef K_ENABLED_ACOS_
void k_acos_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = acosf(A[i]);
}
#endif

#ifdef K_ENABLED_ADD_
void k_add_(float *A, float v, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] += v;
}
#endif

#ifdef K_ENABLED_ASIN_
void k_asin_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = asinf(A[i]);
}
#endif

#ifdef K_ENABLED_ATAN_
void k_atan_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = atanf(A[i]);
}
#endif

#ifdef K_ENABLED_CEIL_
void k_ceil_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ceilf(A[i]);
}
#endif

#ifdef K_ENABLED_CLAMP_
void k_clamp_(float *A, float min, float max, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control
  #pragma HLS INTERFACE s_axilite port=max bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i){
    if (A[i] < min){
      A[i] = min;
    } else if(A[i] > max){
      A[i] = max;
    }
  }
}
#endif

#ifdef K_ENABLED_COS_
void k_cos_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = cosf(A[i]);
}
#endif

#ifdef K_ENABLED_COSH_
void k_cosh_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = coshf(A[i]);
}
#endif

#ifdef K_ENABLED_EXP_
void k_exp_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::expf(A[i]);
}
#endif

#ifdef K_ENABLED_FLOOR_
void k_floor_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::floorf(A[i]);
}
#endif

#ifdef K_ENABLED_INV_
void k_inv_(float *A, float v, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; ++i) A[i] = v/A[i];
}
#endif

#ifdef K_ENABLED_LOG_
void k_log_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::logf(A[i]);
}
#endif

#ifdef K_ENABLED_LOG2_
void k_log2_(float *A, long int size) {
  for (int i = 0; i < size; ++i) A[i] = ::log2f(A[i]);
}
#endif

#ifdef K_ENABLED_LOG10_
void k_log10_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::log10f(A[i]);
}
#endif

#ifdef K_ENABLED_LOGN_
void k_logn_(float *A, float n, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=n bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::logf(A[i])/::logf(n);
}
#endif

#ifdef K_ENABLED_MOD_
void k_mod_(float *A, float v, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::fmod(A[i], v);
}
#endif

#ifdef K_ENABLED_MULT_
void k_mult_(float *A, float v, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] *= v;
}
#endif

#ifdef K_ENABLED_NORMALIZE_
void k_normalize_(float *A, float min, float max, long int size, float Amin, float Amax){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control
  ragma HLS INTERFACE s_axilite port=max bundle=control
  ragma HLS INTERFACE s_axilite port=size bundle=control
  ragma HLS INTERFACE s_axilite port=Amin bundle=control
  #pragma HLS INTERFACE s_axilite port=Amax bundle=control

  // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
  // (max2-min2)/(max1-min1) * (x-min1) + min2
  float max_ori = Amax;
  float min_ori = Amin;
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = (max-min)/(max_ori-min_ori) * (A[i]-min_ori) + min;
}
#endif

#ifdef K_ENABLED_POW_
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
#endif

#ifdef K_ENABLED_POWB_
void k_powb_(float *A, float base, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=base bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    #pragma omp parallel for
    for (int i = 0; i < size; ++i) A[i] = ::powf(base, A[i]);
}
#endif

#ifdef K_ENABLED_RECIPROCAL_
void k_reciprocal_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = 1.0f/A[i];
}
#endif

#ifdef K_ENABLED_REMAINDER_
void k_remainder_(float *A, float v, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=v bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = (int)(A[i]/v);
}
#endif

#ifdef K_ENABLED_ROUND_
void k_round_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::roundf(A[i]);
}
#endif

#ifdef K_ENABLED_RSQRT_
void k_rsqrt_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = 1.0f/::sqrtf(A[i]);
}
#endif

#ifdef K_ENABLED_SIGMOID_
void k_sigmoid_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::expf(A[i])/(::expf(A[i])+1.0f);
}
#endif

#ifdef K_ENABLED_SIGN_
void k_sign_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) {
    if(A[i] > 0.0f){
      A[i] = 1.0f;
    }else if(A[i] < 0.0f){
      A[i] = -1.0f;
    }else{
      A[i] = 0.0f;
    }
  };
}
#endif

#ifdef K_ENABLED_SIN_
void k_sin_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::sinf(A[i]);
}
#endif

#ifdef K_ENABLED_SINH_
void k_sinh_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::sinhf(A[i]);
}
#endif

#ifdef K_ENABLED_SQR_
void k_sqr_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  // pow(x, 2) == x*x  To know more, read comments in pow_'s function
  // speed: 0.000497s
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] *= A[i];
}
#endif

#ifdef K_ENABLED_SQRT_
void k_sqrt_(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::sqrtf(A[i]);
}
#endif

#ifdef K_ENABLED_TAN_
void k_tan_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::tanf(A[i]);
}
#endif

#ifdef K_ENABLED_TANH_
void k_tanh_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::tanhf(A[i]);
}
#endif

#ifdef K_ENABLED_TRUNC_
void k_trunc_(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; ++i) A[i] = ::truncf(A[i]);
}
#endif

#ifdef K_ENABLED_ADD
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

  for (int i = 0; i < size; i++)
    if (incC) C[i] += scA * A[i] + scB * B[i];
    else C[i] = scA * A[i] + scB * B[i];
}
#endif

#ifdef K_ENABLED_INC
void k_inc(float *A, float *B, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    // lock?
  for (int i = 0; i < size; i++)
    B[i] += A[i];
}
#endif

// mult2D should not be a kernel. The multiple ifs run on CPU and the
// proper matrix multiplication kernel is called instead
//void k_mult2D(float *A, int tA, float *B, int tB, float *C, int incC) {
//}

#ifdef K_ENABLED_EL_DIV
void k_el_div(float *A, float *B, float *C, int incC, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=incC bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    if (incC) C[i] += A[i] / B[i];
    else C[i] = A[i] / B[i];
}
#endif

#ifdef K_ENABLED_EL_MULT
void k_el_mult(float *A, float *B, float *C, int incC, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=incC bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
  if (incC) C[i] += A[i] * B[i];
  else C[i] = A[i] * B[i];
}
#endif

#ifdef K_ENABLED_SIGN2
void k_sign2(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
  if (A[i] < 0) B[i] = -1.0;
  else B[i] = 1.0;
}
#endif

#ifdef K_ENABLED_SUM2D_ROWWISE
void k_sum2D_rowwise(float *A, float *B, float *C, int Ashape0, int Ashape1) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control

  for (int i = 0; i < Ashape0; i++) {
    int p=i*Ashape1;
    for (int j = 0; j < Ashape1; j++, p++)
      C[p] = A[p] + B[j];
  }
}
#endif

#ifdef K_ENABLED_SUM2D_COLWISE
void k_sum2D_colwise(float *A, float *B, float *C, int Ashape0, int Ashape1) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=C bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control

  for (int i = 0; i < Ashape0; i++) {
    int p=i*Ashape1;
    for (int j = 0; j < Ashape1; j++, p++)
    C[p] = A[p] + B[i];
  }
}
#endif

#ifdef K_ENABLED_MAX
float k_max(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=max bundle=control //return

  float max = A[0];
  // todo: #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    if (A[i] > max) { max = A[i]; }
  }
}
#endif

#ifdef K_ENABLED_MIN
float k_min(float *A, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=min bundle=control //return

  float min = A[0];
  // todo: #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    if (A[i] < min) { min = A[i]; }
  }
}
#endif

#ifdef K_ENABLED_SUM
float k_sum(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=sum bundle=control //return

  float sum = 0.0;
  for (int i = 0; i < size; ++i) sum += A[i];
}
#endif

#ifdef K_ENABLED_SUM_ABS
float k_sum_abs(float *A, long int size) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=sum bundle=control //return

  float sum = 0.0;
  for (int i = 0; i < size; ++i) sum += ::fabs(A[i]);
}
#endif

}
