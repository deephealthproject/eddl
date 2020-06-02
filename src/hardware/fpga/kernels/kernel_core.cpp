/**********

*******************************************************************************/


//initialize tensor to 'v' value
void fill_(float *A, int Asize, float v){
#pragma HLS INLINE
  for(int i = 0; i<Asize;i++){
        A[i] = v;
  }
}





extern "C" {

void kernel_core(
         float *A,
         int Asize,
	       float v,
         int kernel_id
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=Asize bundle=control
#pragma HLS INTERFACE s_axilite port=v bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  switch (kernel_id) {
    case 20: fill_(A, Asize, v); break;

  }



}
}
