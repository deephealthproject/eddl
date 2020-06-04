#include <math.h>
#include <stdio.h>
extern "C" {
    
void k_abs_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = fabs(A[i]);
}

void k_acos_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = acosf(A[i]);
}

void k_add_(float *A, float v, int size) {
  for (int i = 0; i < size; ++i) A[i] += v;
}

void k_asin_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = asinf(A[i]);
}

void k_atan_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = atanf(A[i]);
}

void k_ceil_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ceilf(A[i]);
}

void k_clamp_(float *A, float min, float max, int size) {
  for (int i = 0; i < size; ++i){
    if (A[i] < min){
      A[i] = min;
    } else if(A[i] > max){
      A[i] = max;
    }
  }
}

void k_cos_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = cosf(A[i]);
}

void k_cosh_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = coshf(A[i]);
}

void k_exp_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::expf(A[i]);
}

void k_floor_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::floorf(A[i]);
}

void k_inv_(float *A, float v, int size) {
    for (int i = 0; i < size; ++i) A[i] = v/A[i];
}

void k_log_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::logf(A[i]);
}

void k_log2_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::log2f(A[i]);
}

void k_log10_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::log10f(A[i]);
}

void k_logn_(float *A, float n, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::logf(A[i])/::logf(n);
}

void k_mod_(float *A, float v, int size){
  for (int i = 0; i < size; ++i) A[i] = ::fmod(A[i], v);
}

void k_mult_(float *A, float v, int size) {
  for (int i = 0; i < size; ++i) A[i] *= v;
}

void k_normalize_(float *A, float min, float max, int size, float Amin, float Amax){
  // Normalize in range: 423 from [23, 562], to range [-1, 1] => 0.4842
  // (max2-min2)/(max1-min1) * (x-min1) + min2
  float max_ori = Amax;
  float min_ori = Amin;
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = (max-min)/(max_ori-min_ori) * (A[i]-min_ori) + min;
}

void k_pow_(float *A, float exp, int size) {
  // To compute the power, std uses real floating-point number with the formurla: e^(y*log_(x))
  // Quite inefficient (x100 slower) in g++ except for pow_(x, 2) which is inlined as x*x
  // speed: 0.057887s
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = ::powf(A[i], exp);
}

void k_powb_(float *A, float base, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) A[i] = ::powf(base, A[i]);
}

void k_reciprocal_(float *A, int size) {
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] = 1.0f/A[i];
}

void k_remainder_(float *A, float v, int size) {
  for (int i = 0; i < size; ++i) A[i] = (int)(A[i]/v);
}

void k_round_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::roundf(A[i]);
}

void k_rsqrt_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = 1.0f/::sqrtf(A[i]);
}

void k_sigmoid_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::expf(A[i])/(::expf(A[i])+1.0f);
}

void k_sign_(float *A, int size){
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

void k_sin_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::sinf(A[i]);
}

void k_sinh_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::sinhf(A[i]);
}

void k_sqr_(float *A, int size) {
  // pow(x, 2) == x*x  To know more, read comments in pow_'s function
  // speed: 0.000497s
  #pragma omp parallel for
  for (int i = 0; i < size; ++i) A[i] *= A[i];
}

void k_sqrt_(float *A, int size) {
  for (int i = 0; i < size; ++i) A[i] = ::sqrtf(A[i]);
}

void k_tan_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::tanf(A[i]);
}

void k_tanh_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::tanhf(A[i]);
}

void k_trunc_(float *A, int size){
  for (int i = 0; i < size; ++i) A[i] = ::truncf(A[i]);
}

void k_add(float scA, float *A, float scB, float *B, float *C, int incC, int size) {
  for (int i = 0; i < size; i++)
    if (incC) C[i] += scA * A[i] + scB * B[i];
    else C[i] = scA * A[i] + scB * B[i];
}

void k_inc(float *A, float *B, int size) {
    // lock?
  for (int i = 0; i < size; i++)
    B[i] += A[i];
}

// mult2D should not be a kernel. The multiple ifs run on CPU and the
// proper matrix multiplication kernel is called instead
//void k_mult2D(float *A, int tA, float *B, int tB, float *C, int incC) {
//}

void k_el_div(float *A, float *B, float *C, int incC, int size) {
  for (int i = 0; i < size; i++)
    if (incC) C[i] += A[i] / B[i];
    else C[i] = A[i] / B[i];
}

void k_el_mult(float *A, float *B, float *C, int incC, int size) {
  for (int i = 0; i < size; i++)
  if (incC) C[i] += A[i] * B[i];
  else C[i] = A[i] * B[i];
}

void k_sign2(float *A, float *B, int size){
  for (int i = 0; i < size; i++)
  if (A[i] < 0) B[i] = -1.0;
  else B[i] = 1.0;
}

void k_sum2D_rowwise(float *A, float *B, float *C, int Ashape0, int Ashape1) {
  for (int i = 0; i < Ashape0; i++) {
    int p=i*Ashape1;
    for (int j = 0; j < Ashape1; j++, p++)
      C[p] = A[p] + B[j];
  }
}

void k_sum2D_colwise(float *A, float *B, float *C, int Ashape0, int Ashape1) {
  for (int i = 0; i < Ashape0; i++) {
    int p=i*Ashape1;
    for (int j = 0; j < Ashape1; j++, p++)
    C[p] = A[p] + B[i];
  }
}

float k_max(float *A, int size){
  float max = FLT_MIN;
  // todo: #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    if (A[i] > max) { max = A[i]; }
  }
}

float k_min(float *A, int size){
  float min = FLT_MAX;
  // todo: #pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    if (A[i] < min) { min = A[i]; }
  }
}

float k_sum(float *A, int size) {
  float sum = 0.0;
  for (int i = 0; i < size; ++i) sum += A[i];
}

float k_sum_abs(float *A, int size) {
  float sum = 0.0;
  for (int i = 0; i < size; ++i) sum += ::fabs(A[i]);
}

}

