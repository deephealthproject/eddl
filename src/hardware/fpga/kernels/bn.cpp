#include <math.h>
#include <stdio.h>
extern "C" {

void k_permute_channels_last(float *A,float *B, int Ashape0, int Ashape1, int Ashape2, int Ashape3)
{
  int b,z,r,c;

  b=Ashape0;
  z=Ashape1;
  r=Ashape2;
  c=Ashape3;

  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=i*(r*c*z)+k*(c*z)+m*z+j;
          B[pdest]=A[psrc];
        }
  }
}

void k_permute_channels_first(float *A,float *B, int Bshape0, int Bshape1, int Bshape2, int Bshape3)
{
  int b,z,r,c;

  b=Bshape0;
  z=Bshape1;
  r=Bshape2;
  c=Bshape3;

  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=i*(r*c*z)+k*(c*z)+m*z+j;
          B[psrc]=A[pdest];
        }
  }
}

void k_permute_batch_last(float *A,float *B, int Ashape0, int Ashape1, int Ashape2, int Ashape3)
{
  int b,z,r,c;

  b=Ashape0;
  z=Ashape1;
  r=Ashape2;
  c=Ashape3;

  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=j*(r*c*b)+k*(c*b)+m*b+i;
          B[pdest]=A[psrc];
        }
  }
}

void k_permute_batch_first(float *A,float *B, int Bshape0, int Bshape1, int Bshape2, int Bshape3)
{
  int b,z,r,c;

  b=Bshape0;
  z=Bshape1;
  r=Bshape2;
  c=Bshape3;

  #pragma omp parallel for
  for (int i = 0; i < b; ++i) {
    int psrc=i*(z*r*c);
    for(int j=0;j<z;j++)
      for(int k=0;k<r;k++)
        for(int m=0;m<c;m++,psrc++) {
          int pdest=j*(r*c*b)+k*(c*b)+m*b+i;
          B[psrc]=A[pdest];
        }
  }
}

}