/**********

*******************************************************************************/

#include <math.h>

extern "C" {

void kernel_normalize(
         float *A, 
         float max, 
         float min, 
         float max_ori,
         float min_ori,
         int tam 
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=max bundle=control
#pragma HLS INTERFACE s_axilite port=min bundle=control
#pragma HLS INTERFACE s_axilite port=max_ori bundle=control
#pragma HLS INTERFACE s_axilite port=min_ori bundle=control
#pragma HLS INTERFACE s_axilite port=tam bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
       
for (int i = 0; i < tam; ++i) A[i] = (max-min)/(max_ori-min_ori) * (A[i]-min_ori) + min;


}
} 
