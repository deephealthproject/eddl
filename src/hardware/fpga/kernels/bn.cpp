#include <math.h>
#include <stdio.h>
extern "C" {

#ifdef K_ENABLED_PERMUTE_CHANNELS_LAST
void k_permute_channels_last(float *A,float *B, int Ashape0, int Ashape1, int Ashape2, int Ashape3)
{
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape2 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape3 bundle=control

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
#endif

#ifdef K_ENABLED_PERMUTE_CHANNELS_FIRST
void k_permute_channels_first(float *A,float *B, int Bshape0, int Bshape1, int Bshape2, int Bshape3)
{
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape2 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape3 bundle=control

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
#endif

#ifdef K_ENABLED_PERMUTE_BATCH_LAST
void k_permute_batch_last(float *A,float *B, int Ashape0, int Ashape1, int Ashape2, int Ashape3)
{
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape2 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape3 bundle=control

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
#endif

#ifdef K_ENABLED_PERMUTE_BATCH_FIRST
void k_permute_batch_first(float *A,float *B, int Bshape0, int Bshape1, int Bshape2, int Bshape3)
{
  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape2 bundle=control
  #pragma HLS INTERFACE s_axilite port=Bshape3 bundle=control
  
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
#endif

}
