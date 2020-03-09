/**********

*******************************************************************************/


//initialize tensor to 'v' value
void fill_(float *A, int dim0, int dim1, float v){
#pragma HLS INLINE
  for(int i = 0; i<dim0;i++){
      for(int j = 0; j<dim1;j++){
        A[i*dim1 +j] = v;
      }
    }
}





extern "C" {

void kernel_core(
         float *A,
         int dim0,
         int dim1,
	       float v,
         int kernel_id
        )
{

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=v bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  switch (kernel_id) {
    case 20: fill_(A, dim0, dim1, v); break;

  }



}
}
