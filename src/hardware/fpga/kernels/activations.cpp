#include <math.h>
#include <stdio.h>
extern "C" {

void k_relu(float *A, float *B, int size){
}

void k_d_relu(float *D, float *I, float *PD, int size) {
}

void k_thresholded_relu(float *A, float *B, int size, float param){
}

void k_d_thresholded_relu(float *D, float *I, float *PD, int size, float param){
}

void k_leaky_relu(floar *A, float *B, int size, float param){
}

void k_d_leaky_relu(float *D, float *I, float *PD, int size, float param){
}

void k_elu(float *A, float *B, int size, float param){
}

void k_d_elu(float *D, float *I, float *PD, int size, float param){
}

void k_softplus(float *A, float *B, int size){
}

void k_d_softplus(float *D, float *I, int size, float *PD){
}

void k_softsign(float *A, float *B, int size){
}

void k_d_softsign(float *D, float *I, float *PD, int size) {
}

void k_linear(float *A, float *B, int size, float param){
}

void k_d_linear(float *D, float *I, float *PD, float param){
  }

void k_sigmoid(float *A, float *B){
}

void k_d_sigmoid(float *D, float *I, float *PD){
}

void k_hard_sigmoid(float *A, float *B){
}

void k_d_hard_sigmoid(float *D, float *I, float *PD){
}

void k_exp(float *A, float *B){
}

void k_d_exp(float *D, float *I, float *PD){
}

void k_tanh(float *A, float *B){
}

void k_d_tanh(float *D, float *I, float *PD){
}

void k_softmax(float *A, float *B) {
}

void k_d_softmax(float *D, float *I, float *PD) {
}

} // end extern "C"