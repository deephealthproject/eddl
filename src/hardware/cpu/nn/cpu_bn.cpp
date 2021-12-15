/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"

#include "eddl/hardware/cpu/cpu_tensor.h"

// BN
void cpu_permute_channels_last(Tensor *A,Tensor *B)
{
  _profile(_CPU_PERMUTE_CHANELS_LAST, 0);
  int b,z,r,c;

  b=A->shape[0];
  z=A->shape[1];
  r=A->shape[2];
  c=A->shape[3];

  #pragma omp parallel for
  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=i*(r*c*z)+k*(c*z)+m*z+j;
          B->ptr[pdest]=A->ptr[psrc];
        }
  }
    _profile(_CPU_PERMUTE_CHANELS_LAST, 1);

}

void cpu_permute_channels_first(Tensor *A,Tensor *B)
{
    _profile(_CPU_PERMUTE_CHANELS_FIRST, 0);
  int b,z,r,c;

  b=B->shape[0];
  z=B->shape[1];
  r=B->shape[2];
  c=B->shape[3];


  #pragma omp parallel for
  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=i*(r*c*z)+k*(c*z)+m*z+j;
          B->ptr[psrc]=A->ptr[pdest];
        }
  }
    _profile(_CPU_PERMUTE_CHANELS_FIRST, 1);

}

void cpu_permute_batch_last(Tensor *A,Tensor *B)
{
  _profile(_CPU_PERMUTE_BATCH_LAST, 0);
  int b,z,r,c;

  b=A->shape[0];
  z=A->shape[1];
  r=A->shape[2];
  c=A->shape[3];

  #pragma omp parallel for
  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=j*(r*c*b)+k*(c*b)+m*b+i;
          B->ptr[pdest]=A->ptr[psrc];
        }
  }
    _profile(_CPU_PERMUTE_BATCH_LAST, 1);

}

void cpu_permute_batch_first(Tensor *A,Tensor *B)
{
  _profile(_CPU_PERMUTE_BATCH_FIRST, 0);
  int b,z,r,c;

  b=B->shape[0];
  z=B->shape[1];
  r=B->shape[2];
  c=B->shape[3];


  #pragma omp parallel for
  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=j*(r*c*b)+k*(c*b)+m*b+i;
          B->ptr[psrc]=A->ptr[pdest];
        }
  }
    _profile(_CPU_PERMUTE_BATCH_FIRST, 1);

}

void cpu_batchnorm_forward(int b, int z, int rc,
        float *input, float *output, float *opa,
        float *global_mean, float *global_variance,
        float *affine_g, float *affine_b,
        float *mean, float *variance,
        bool trmode, float epsilon, float momentum)
{
    const int block_size = 256;
    int rcz = rc * z;
    if (trmode) {
        // compute mean and variance
        for (int j = 0; j < z; j++) mean[j] = variance[j] = 0.0;
        // warning: using omp in the next loop can lead to unstable computations of mean and variance
        // gpu version protects it with atomicAdd()
        #pragma omp parallel for
        for (int k = 0; k < rcz; k += block_size)
            for (int i = 0; i < b; i++) {
                int p = k + i * rcz;
                for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                    int j = (k + l) / rc;
                    mean[j] += input[p];
                    variance[j] += input[p] * input[p];
                }
            }
        float N = b * rc;
        #pragma omp parallel for
        for (int j = 0; j < z; j++) {
            mean[j] = mean[j] / N;
            variance[j] = variance[j] / N - mean[j] * mean[j];
            // update global statistics
            if (momentum != 0.0) {
                global_mean[j] = momentum * global_mean[j] + (1.0 - momentum) * mean[j];
                global_variance[j] = momentum * global_variance[j] + (1.0 - momentum) * variance[j];
            }
            variance[j] = sqrt(variance[j] + epsilon);
        }
    } else {
        // just update variance from the global variance if momentum is != 0.0,
        // otherwise the mean and variance of the current batch are used, which are
        // computed in the previous block, that will be executed if in TRMODE or momemtum is zero
        mean = global_mean;
        #pragma omp parallel for
        for (int j = 0; j < z; j++) {
            variance[j] = sqrt(global_variance[j] + epsilon);
        }
    }
    // normalization
    #pragma omp parallel for
    for (int k = 0; k < rcz; k += block_size)
        for (int i = 0; i < b; i++) {
            int p = k + i * rcz;
            for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                int j = (k + l) / rc;
                float o = (input[p] - mean[j]) / variance[j];
                if (affine_g != nullptr) {
                    opa[p] = o;
                    output[p] = o * affine_g[j] + affine_b[j]; // apply affine transformation
                } else {
                    output[p] = o;
                }
            }
        }
}

void cpu_batchnorm_backward(int b, int z, int rc,
                            float *delta, float *opa, float *pdelta,
                            float *gbn_g, float *gbn_b, float *bn_g,
                            float *variance, float *mean1, float *mean2)
{
    const int block_size = 256;
    int rcz = rc * z;
    float N = b * rc;
    if (bn_g != NULL) { // affine
        // compute mean
        for (int j = 0; j < z; j++) mean1[j] = mean2[j] = 0.0;
        // warning: using omp in the next loop can lead to unstable computations of mean and variance
        // gpu version protects it with atomicAdd()
        #pragma omp parallel for
        for (int k = 0; k < rcz; k += block_size)
            for (int i = 0; i < b; i++) {
                int p = k + i * rcz;
                for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                    int j = (k + l) / rc;
                    mean1[j] += delta[p] * opa[p];
                    mean2[j] += delta[p];
                    delta[p] *= bn_g[j];
                }
            }
        #pragma omp parallel for
        for (int j = 0; j < z; j++) {
            mean1[j] /= N;
            mean2[j] /= N;
            gbn_g[j] += mean1[j];
            gbn_b[j] += mean2[j];
            mean1[j] *= bn_g[j];
            mean2[j] *= bn_g[j];
        }
    } else {
        // compute mean
        for (int j = 0; j < z; j++) mean1[j] = mean2[j] = 0.0;
        // warning: using omp in the next loop can lead to unstable computations of mean and variance
        // gpu version protects it with atomicAdd()
        #pragma omp parallel for
        for (int k = 0; k < rcz; k += block_size)
            for (int i = 0; i < b; i++) {
                int p = k + i * rcz;
                for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                    int j = (k + l) / rc;
                    mean1[j] += delta[p] * opa[p]; // step 1 & 2
                    mean2[j] += delta[p]; // step 4
                }
            }
        #pragma omp parallel for
        for (int j = 0; j < z; j++) {
            mean1[j] /= N;
            mean2[j] /= N;
        }
    }
    #pragma omp parallel for
    for (int k = 0; k < rcz; k += block_size)
        for (int i = 0; i < b; i++) {
            int p = k + i * rcz;
            for (int l = 0; l < block_size && k + l < rcz; l++, p++) {
                int j = (k + l) / rc;
                // opa[p] = opa[p] * mean1[j] + mean2[j]; // step 3 & 5
                // delta[p] -= opa[p]; // step 6
                // delta[p] /= variance[j]; // step 7
                // pdelta[p] += delta[p];
                pdelta[p] += (delta[p] - (opa[p] * mean1[j] + mean2[j])) / variance[j];
            }
        }
}
