#include <math.h>
#include <stdio.h>

extern "C" {

void k_relu(float *A, float *B, int size){
  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = 0.0;
  }
}

void k_d_relu(float *D, float *I, float *PD, int size) {
  for (int i = 0; i < size; i++) {
    if (I[i] > 0.0) PD[i] += D[i];  // why += ?
    else PD[i] += 0.0;
  }
}

void k_thresholded_relu(float *A, float *B, int size, float param){
  for (int i = 0; i < size; i++) {
    if (A[i] > param) B[i] = A[i];
    else B[i] = 0.0;
  }
}

void k_d_thresholded_relu(float *D, float *I, float *PD, int size, float param){
  for (int i = 0; i < size; i++) {
    if (I[i] > param) PD[i] += D[i];  // why += ?
    else PD[i] += 0.0;
  }
}

void k_leaky_relu(float *A, float *B, int size, float param){
  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = param*A[i];
  }
}

void k_d_leaky_relu(float *D, float *I, float *PD, int size, float param){
  for (int i = 0; i < size; i++) {
    if (I[i] > 0.0) PD[i] += D[i];  // why += ?
    else PD[i] += param*D[i];
  }
}

void k_elu(float *A, float *B, int size, float param){
  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = param * (expf(A[i]) - 1.0);  // check expf is ok
  }
}

void k_d_elu(float *D, float *I, float *PD, int size, float param){
  for (int i = 0; i < size; i++) {
    if (I[i] > 0.0) PD[i] += D[i];  // why +=
    else PD[i] += D[i] * (param * expf(I[i]));  // check expf is ok
  }

}

void k_softplus(float *A, float *B, int size){
    for (int i = 0; i < size; i++) {
        B[i] = logf(1 + expf(A[i]));  // check logf, expf
    }
}

void k_d_softplus(float *D, float *I, int size, float *PD){
    for (int i = 0; i < size; i++) {
        PD[i] += D[i] * 1/(1 + (-I[i]));  // why +=
    }
}

void k_softsign(float *A, float *B, int size){
    for (int i = 0; i < size; i++) {
        B[i] = A[i] / (1 + fabs(A[i]));  // check fabs
    }
}

void k_d_softsign(float *D, float *I, float *PD, int size) {
    for (int i = 0; i < size; i++) {
        float denom = 1 + fabs(I[i]);  // check fabs
        PD[i] += D[i] * 1/(denom*denom);  // why +=
    }
}

void k_linear(float *A, float *B, float param, int size){
  for (int i = 0; i < size; i++) {
    B[i] = param * A[i];
  }
}

void k_d_linear(float *D, float *I, float *PD, float param, int size){
  for (int i = 0; i < size; i++) {
    PD[i] += D[i] * param;
  }
}

void k_sigmoid(float *A, float *B, int size){
  for (int i = 0; i < size; i++)
    B[i] = 1/(1+exp(-A[i]));  // check exp
}

void k_d_sigmoid(float *D, float *I, float *PD, int size){
  for (int i = 0; i < size; i++)
    PD[i] += D[i]*((1-I[i])*I[i]);
}

void k_hard_sigmoid(float *A, float *B, int size){
  for (int i = 0; i < size; i++) {
    if (A[i] > 2.5) B[i] = 1.0;
    else if (A[i] < -2.5) B[i] = 0.0;
    else B[i] = (0.2 * A[i]) + 0.5;
  }
}

void k_d_hard_sigmoid(float *D, float *I, float *PD, int size){
  for (int i = 0; i < size; i++)
    if (I[i] < -2.5 || I[i] > 2.5) PD[i] += 0;
    else PD[i] += D[i] * 0.2;
}

void k_exp(float *A, float *B, int size){
  for (int i = 0; i < size; i++) {
    B[i] = exp(A[i]);
  }
}

void k_d_exp(float *D, float *I, float *PD, int size){
  for (int i = 0; i < size; i++)
    PD[i] += D[i] * I[i];
}

void k_tanh(float *A, float *B, int size){
  for (int i = 0; i < size; i++) {
    float p=exp(A[i]);
    float n=exp(-A[i]);
    B[i] = (p-n)/(p+n);
  }
}

void k_d_tanh(float *D, float *I, float *PD, int size){
  for (int i = 0; i < size; i++)
    PD[i] += D[i]*(1-(I[i]*I[i]));
}

void k_softmax(float *A, float *B, int Ashape0, int Ashape1, int Bshape1) {

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

void k_d_softmax(float *D, float *I, float *PD, int size) {
  for (int i = 0; i < size; i++)
    PD[i] += D[i] * (I[i] * (1.0 - I[i]));
}

} // end extern "C"