/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
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

}

void cpu_permute_channels_first(Tensor *A,Tensor *B)
{
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

}

void cpu_permute_batch_last(Tensor *A,Tensor *B)
{
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

}

void cpu_permute_batch_first(Tensor *A,Tensor *B)
{
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

}
