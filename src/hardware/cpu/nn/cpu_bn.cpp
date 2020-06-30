/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.7
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>      /* printf, scanf, NULL */
#include <cstdlib>     /* malloc, free, rand */
#include <iostream>

#include "eddl/hardware/cpu/nn/cpu_tensor_nn.h"


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
