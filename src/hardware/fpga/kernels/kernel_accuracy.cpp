/**********

*******************************************************************************/

#include <math.h>


extern "C" {

void kernel_accuracy(
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
      
      for(int i=0;i<dim0;i++){ // get the max in the column
          for (int j=0; j< dim1; j++){     
                if (A[dim0*j+i]>aind) aind = A[dim0*j+i];
                if (B[dim0*j+i]>bind) bind = B[dim0*j+i];
          } 
          if (aind==bind) *acc++;
      }



}
} 
