#include <math.h>
#include <stdio.h>


extern "C" {

#ifdef K_ENABLED_RELU
void k_relu(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control


  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = 0.0;
  }
}
#endif

#ifdef K_ENABLED_D_RELU
void k_d_relu(float *D, float *I, float *PD, long int size) {

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++) {
    if (I[i] > 0.0) PD[i] += D[i];  // why += ?
    else PD[i] += 0.0;
  }
}
#endif

#ifdef K_ENABLED_THRESHOLDED_RELU
void k_thresholded_relu(float *A, float *B, long int size, float param){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > param) B[i] = A[i];
    else B[i] = 0.0;
  }
}

#endif

#ifdef K_ENABLED_D_TRHESHOLDED_RELU
void k_d_thresholded_relu(float *D, float *I, float *PD, long int size, float param){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (I[i] > param) PD[i] += D[i];  // why += ?
    else PD[i] += 0.0;
  }
}
#endif

#ifdef K_ENABLED_LEAKY_RELU
void k_leaky_relu(float *A, float *B, long int size, float param){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = param*A[i];
  }
}
#endif

#ifdef K_ENABLED_D_LEAKY_RELU
void k_d_leaky_relu(float *D, float *I, float *PD, long int size, float param){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (I[i] > 0.0) PD[i] += D[i];  // why += ?
    else PD[i] += param*D[i];
  }
}
#endif

#ifdef K_ENABLED_ELU
void k_elu(float *A, float *B, long int size, float param){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > 0.0) B[i] = A[i];
    else B[i] = param * (expf(A[i]) - 1.0);  // check expf is ok
  }
}
#endif

#ifdef K_ENABLED_D_ELU
void k_d_elu(float *D, float *I, float *PD, long int size, float param){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    if (I[i] > 0.0) PD[i] += D[i];  // why +=
    else PD[i] += D[i] * (param * expf(I[i]));  // check expf is ok
  }

}
#endif

#ifdef K_ENABLED_SOFTPLUS
void k_softplus(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        B[i] = logf(1 + expf(A[i]));  // check logf, expf
    }
}
#endif

#ifdef K_ENABLED_D_SOFTPLUS
void k_d_softplus(float *D, float *I, long int size, float *PD){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        PD[i] += D[i] * 1/(1 + (-I[i]));  // why +=
    }
}
#endif

#ifdef K_ENABLED_SOFTSIGN
void k_softsign(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        B[i] = A[i] / (1 + fabs(A[i]));  // check fabs
    }
}
#endif

#ifdef K_ENABLED_D_SOFTSIGN
void k_d_softsign(float *D, float *I, float *PD, long int size) {

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

    for (int i = 0; i < size; i++) {
        float denom = 1 + fabs(I[i]);  // check fabs
        PD[i] += D[i] * 1/(denom*denom);  // why +=
    }
}
#endif

#ifdef K_ENABLED_LINEAR
void k_linear(float *A, float *B, float param, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    B[i] = param * A[i];
  }
}
#endif

#ifdef K_ENABLED_D_LINEAR
void k_d_linear(float *D, float *I, float *PD, float param, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  #pragma HLS INTERFACE s_axilite port=param bundle=control

  for (int i = 0; i < size; i++) {
    PD[i] += D[i] * param;
  }
}
#endif

#ifdef K_ENABLED_SIGMOID
void k_sigmoid(float *A, float *B, long int size){
  for (int i = 0; i < size; i++)
    B[i] = 1/(1+exp(-A[i]));  // check exp
}
#endif

#ifdef K_ENABLED_D_SIGMOID
void k_d_sigmoid(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    PD[i] += D[i]*((1-I[i])*I[i]);
}
#endif

#ifdef K_ENABLED_HARD_SIGMOID
void k_hard_sigmoid(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++) {
    if (A[i] > 2.5) B[i] = 1.0;
    else if (A[i] < -2.5) B[i] = 0.0;
    else B[i] = (0.2 * A[i]) + 0.5;
  }
}
#endif

#ifdef K_ENABLED_D_HARD_SIGMOID
void k_d_hard_sigmoid(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    if (I[i] < -2.5 || I[i] > 2.5) PD[i] += 0;
    else PD[i] += D[i] * 0.2;
}
#endif

#ifdef K_ENABLED_EXP
void k_exp(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++) {
    B[i] = exp(A[i]);
  }
}
#endif

#ifdef K_ENABLED_D_EXP
void k_d_exp(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    PD[i] += D[i] * I[i];
}
#endif

#ifdef K_ENABLED_TANH
void k_tanh(float *A, float *B, long int size){

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++) {
    float p=exp(A[i]);
    float n=exp(-A[i]);
    B[i] = (p-n)/(p+n);
  }
}
#endif

#ifdef K_ENABLED_D_TANH
void k_d_tanh(float *D, float *I, float *PD, long int size){

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    PD[i] += D[i]*(1-(I[i]*I[i]));
}
#endif

#ifdef K_ENABLED_SOFTMAX
void k_softmax(float *A, float *B, int Ashape0, int Ashape1, int Bshape1) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape1 bundle=control

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
#endif

#ifdef K_ENABLED_D_SOFTMAX
void k_d_softmax(float *D, float *I, float *PD, long int size) {

  #pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=D  bundle=control
  #pragma HLS INTERFACE s_axilite port=I  bundle=control
  #pragma HLS INTERFACE s_axilite port=PD  bundle=control
  #pragma HLS INTERFACE s_axilite port=size bundle=control

  for (int i = 0; i < size; i++)
    PD[i] += D[i] * (I[i] * (1.0 - I[i]));
}
#endif

} // end extern "C"
