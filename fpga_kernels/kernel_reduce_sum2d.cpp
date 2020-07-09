#include <math.h>
#include <stdio.h>
extern "C" {

void k_reduce_sum2d(float *A, float *B, int Ashape0, int Ashape1, int axis, int incB) {

  #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  #pragma HLS INTERFACE s_axilite port=A  bundle=control
  #pragma HLS INTERFACE s_axilite port=B  bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape0 bundle=control
  #pragma HLS INTERFACE s_axilite port=Ashape1 bundle=control
  #pragma HLS INTERFACE s_axilite port=axis bundle=control
  #pragma HLS INTERFACE s_axilite port=incB bundle=control
  #pragma HLS INTERFACE s_axilite port=return bundle=control


    if (axis==0){
        if (!incB) for(int i=0;i<Ashape1;++i) {B[i]=0;}
        int p=0;
        for(int i=0;i<Ashape0;++i) {
          for(int j=0;j<Ashape1;++j,p++){
              B[j]+=A[p];
          }
        }
     }
     else{
        if (!incB) for(int i=0;i<Ashape0;++i){B[i]=0;}
        int p=0;
        for(int i=0;i<Ashape0;++i) {
          for(int j=0;j<Ashape1;++j,p++){
                B[i]+=A[p];
          }
        }
    }
}

} // end extern "C"
