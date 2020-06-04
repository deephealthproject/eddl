/**********

*******************************************************************************/

#include <math.h>
#include <stdio.h>

extern "C" {

void k_accuracy(
         const float *A,
         const float *B,
         int dim0,
         int dim1,
	       int *acc
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE m_axi port=acc offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=acc bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

      int aind = 0,bind = 0;
      *acc = 0;
      int accuracy = 0;
      float maxA = 0;
      float maxB = 0;

      for(int i=0;i<dim0;i++){ 
        maxA = A[i*dim1];
        maxB = B[i*dim1];

          for (int j=0; j< dim1; j++){
                if (A[dim1*i+j]>maxA){
                  aind = j;
                  maxA = A[dim1*i+j];
                }
                else aind = aind;
                if (B[dim1*i+j]>maxB){
                   bind = j;
                   maxB = B[dim1*i+j];
                 }
                else bind=bind;
          }
          // printf("max [%d] = %f \n",i,maxA);
          // printf("max [%d] = %f \n",i,maxB);
          // printf("max aind [%d] = %d \n",i,aind);
          // printf("max bind [%d] = %d \n",i,bind);
          if (aind==bind) accuracy++;
      }
      //printf("acc_kernel = %d \n", accuracy);
      *acc = accuracy;

}
}
