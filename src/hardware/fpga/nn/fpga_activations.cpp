/*
* FPGA support for EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), GAP research group
* Date: June 2020
* Author: GAP Research Group (UPV), contact: carlherlu@gap.upv.es, jflich@disca.upv.es
* All rights reserved
*/



#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/fpga/nn/fpga_nn.h"

void fpga_relu(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_RELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_RELU, 1);
}

void fpga_d_relu(Tensor *D, Tensor *I, Tensor *PD){
 _profile_fpga(_FPGA_D_RELU, 0);
 printf("fpga_ not implemented yet\n"); exit(1);
 _profile_fpga(_FPGA_D_RELU, 1);
}

void fpga_thresholded_relu(Tensor *A, Tensor *B,float param){
  _profile_fpga(_FPGA_THRESHOLDED_RELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_THRESHOLDED_RELU, 1);
}

void fpga_d_thresholded_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
  _profile_fpga(_FPGA_D_THRESHOLDED_RELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_THRESHOLDED_RELU, 1);
}

void fpga_leaky_relu(Tensor *A, Tensor *B,float param){
  _profile_fpga(_FPGA_LEAKY_RELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_LEAKY_RELU, 1);
}

void fpga_d_leaky_relu(Tensor *D, Tensor *I, Tensor *PD,float param){
  _profile_fpga(_FPGA_D_LEAKY_RELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_LEAKY_RELU, 1);
}

void fpga_elu(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_ELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_ELU, 1);
}

void fpga_d_elu(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_ELU, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_ELU, 1);
}

void fpga_softplus(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_SOFTPLUS, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SOFTPLUS, 1);
}

void fpga_d_softplus(Tensor *D, Tensor *I, Tensor *PD){
    _profile_fpga(_FPGA_D_SOFTPLUS, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_D_SOFTPLUS, 1);
}

void fpga_softsign(Tensor *A, Tensor *B){
    _profile_fpga(_FPGA_SOFTSIGN, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_SOFTSIGN, 1);
}

void fpga_d_softsign(Tensor *D, Tensor *I, Tensor *PD){
    _profile_fpga(_FPGA_D_SOFTSIGN, 0);
    printf("fpga_ not implemented yet\n"); exit(1);
    _profile_fpga(_FPGA_D_SOFTSIGN, 1);
}

void fpga_linear(Tensor *A, Tensor *B, float param){
  _profile_fpga(_FPGA_LINEAR, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_LINEAR, 1);
}

void fpga_d_linear(Tensor *D, Tensor *I, Tensor *PD, float param){
  _profile_fpga(_FPGA_D_LINEAR, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_LINEAR, 1);
}

void fpga_sigmoid(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_SIGMOID, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SIGMOID, 1);
}

void fpga_d_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_SIGMOID, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_SIGMOID, 1);
}

void fpga_hard_sigmoid(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_HARD_SIGMOID, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_HARD_SIGMOID, 1);
}

void fpga_d_hard_sigmoid(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_HARD_SIGMOID, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_HARD_SIGMOID, 1);
}

void fpga_exp(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_EXP, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_EXP, 1);
}

void fpga_d_exp(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_EXP, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_EXP, 1);
}

void fpga_tanh(Tensor *A, Tensor *B){
  _profile_fpga(_FPGA_TANH, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_TANH, 1);
}

void fpga_d_tanh(Tensor *D, Tensor *I, Tensor *PD){
  _profile_fpga(_FPGA_D_TANH, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_D_TANH, 1);
}

void fpga_softmax(Tensor *A, Tensor *B) {
  _profile_fpga(_FPGA_SOFTMAX, 0);
  printf("fpga_ not implemented yet\n"); exit(1);
  _profile_fpga(_FPGA_SOFTMAX, 1);
}

void fpga_d_softmax(Tensor *D, Tensor *I, Tensor *PD) {
    _profile_fpga(_FPGA_D_SOFTMAX, 0);
  PD->tsem->lock();
  printf("fpga_ not implemented yet\n"); exit(1);
  PD->tsem->unlock();
  _profile_fpga(_FPGA_D_SOFTMAX, 1);
}
