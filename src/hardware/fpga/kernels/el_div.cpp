/**********

*******************************************************************************/

#include <math.h>
#include <stdio.h>
extern "C" {

void k_el_div(
         const float *A, 
         const float *B,
         float *C, 
         int size, 
         int incC, 
         int op
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=incC bundle=control
#pragma HLS INTERFACE s_axilite port=op bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  printf("EL div\n");

  if (op == 0) { //DIV
     for (int i = 0; i < size; i++){
         if (incC) C[i] += A[i] / B[i];
         else C[i] = A[i] / B[i];
     }
  }else{
     for(int i=0; i<size; i++){
         if (incC) C[i]+=A[i]*B[i];
         else C[i]=A[i]*B[i];
     }
  }


}
} 
