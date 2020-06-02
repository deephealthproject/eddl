/**********

*******************************************************************************/

#include <math.h>
#include <stdio.h>
extern "C" {

void reduce_sum2D(
        float *A, // Output Tensor
        float *B,
        int dim0, 
        int dim1,
        int axis,
        int incB 
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=axis bundle=control
#pragma HLS INTERFACE s_axilite port=incB bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    printf("In reduce SUM\n");   
    /*if (axis==0)
      {
        if (!incB) for(int i=0;i<dim1;++i) {B[i]=0; printf("i %d dim0 %d dim1 %d\n", i, dim0, dim1);};

        int p=0;
        for(int i=0;i<dim0;++i) {
          for(int j=0;j<dim1;++j,p++){
               printf("j %d p %d dim0 %d dim1 %d\n", j,p, dim0, dim1);    
               B[j]+=A[p];
          }
        }  
     }else
        {
          if (!incB) for(int i=0;i<dim0;++i) B[i]=0;

          int p=0;
          for(int i=0;i<dim0;++i) {
            for(int j=0;j<dim1;++j,p++){
                 printf("j %d p %d dim0 %d dim1 %d\n", j,p, dim0, dim1);
                 B[i]+=A[p];
            }
          }
     }
*/
}
} 
