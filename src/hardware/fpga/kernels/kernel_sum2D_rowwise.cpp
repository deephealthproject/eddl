/**********

*******************************************************************************/

#include <math.h>

extern "C" {

void kernel_sum2D_rowwise(
         const float *A, 
         const float *B, 
         float *C,
         int dim0, 
         int dim1
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

      int p=0;

      for(int i=0;i<dim0;i++) {
        for(int j=0;j<dim1;j++,p++)
          C[p]= A[p]+B[j];
      }

}
} 
