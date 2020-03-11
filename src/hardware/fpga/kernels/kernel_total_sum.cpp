/**********

*******************************************************************************/

#include <math.h>
#include <stdio.h>

extern "C" {

void kernel_total_sum(
         const float *A,
         int Asize,
	       float *sum
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=Asize bundle=control
#pragma HLS INTERFACE m_axi port=sum offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=sum bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
      float local_sum = 0.0;
      for(int i=0;i<Asize;i++){ // get the max in the column
         local_sum += A[i];
      }

     *sum = local_sum;


}
}
