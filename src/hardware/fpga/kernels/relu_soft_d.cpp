/**********

*******************************************************************************/

#include <math.h>

void k_relu_d(const float *D, const float *I, float *PD, int tam){
#pragma HLS INLINE   
    for(int i=0;i<tam;i++) {
       if (I[i]>0.0) PD[i]=D[i];
       else PD[i]=0.0;
    } 
}

void k_softmax_d(const float *D, const float *I, float *PD, int tam){
#pragma HLS INLINE
    for(int i=0;i<tam;i++)
         PD[i]+=D[i]*(I[i]*(1.0-I[i]));
}

extern "C" {

void k_relu_soft_d(
        const float *D, // Output Tensor
        const float *I,
        float *PD,
        int tam,        
        int kernel_id
        )
{
#pragma HLS INLINE
#pragma HLS INTERFACE m_axi port=D offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=I offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=PD offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=D  bundle=control
#pragma HLS INTERFACE s_axilite port=I  bundle=control
#pragma HLS INTERFACE s_axilite port=PD  bundle=control
#pragma HLS INTERFACE s_axilite port=tam bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
   switch (kernel_id) {
       case 9: k_relu_d(D, I, PD, tam); break;
       case 10: k_softmax_d(D, I, PD, tam); break;
   }

}


}
