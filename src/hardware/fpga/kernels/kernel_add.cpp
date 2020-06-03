/**********

*******************************************************************************/

#include <math.h>
#define MAX_SIZE 1024*1024


extern "C" {

void k_add(
         float scA,
         const float *A,
         float scB,
         const float *B,
         float *C,
         int incC,
         int tam
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=B  bundle=control
#pragma HLS INTERFACE s_axilite port=C  bundle=control
#pragma HLS INTERFACE s_axilite port=scA bundle=control
#pragma HLS INTERFACE s_axilite port=scB bundle=control
#pragma HLS INTERFACE s_axilite port=incC bundle=control
#pragma HLS INTERFACE s_axilite port=tam bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control


for(int i=0;i<tam;i++){
      if (incC) C[i]+=scA*A[i]+scB*B[i];
      else C[i]=scA*A[i]+scB*B[i];
}


}
}
