/**********

*******************************************************************************/

#include <math.h>

extern "C" {

void kernel_cent(
         const float *A, 
         const float *B, 
         float *C,
         int tam 
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=tam bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    for(int i=0;i<tam;i++) {
       C[i]=0;
       if (A[i]!=0.0) C[i]-=A[i]*log(B[i]);
       if (A[i]!=1.0) C[i]-=(1.0-A[i])*log(1.0-B[i]);
    }

}
} 
