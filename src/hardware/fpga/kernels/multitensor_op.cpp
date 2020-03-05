/**********

*******************************************************************************/

#include <math.h>

void relu(const float *A, float *B, int tam){
#pragma HLS INLINE          
     for(int i=0;i<tam;i++) {
        if (A[i]>0.0) B[i]=A[i];
        else B[i]=0.0;
     }
}

void softmax(const float *A, float *B, int dim0, int dim1){
#pragma HLS INLINE
   float max;
   float sum;

   for(int i = 0; i<dim0;i++){
     max = A[i];
     for(int j = 0; j<dim1;j++){
       if(A[i*dim1 +j]>max){
         max = A[i*dim1 +j];
       }
     }
    //printf("max[%d] = %f\n",i, max);


  //e^(A[i][j]-max)
  //sum = sum(B_pre[i][j]);
    sum = 0;
    for(int j = 0; j<dim1;j++){
      B[i*dim1+j] = exp(A[i*dim1+j]-max);
      // printf("MAX_nextloop = %f\n",max);
      // printf("B_pre[%d][%d]: %f\n", i,j,  B[i*dim1+j]);
      sum = sum + B[i*dim1+j];
    }


    for (int j=0; j < dim1; j++) {
      // printf("sum[%d] = %f\n", j , sum);
      B[i*dim1+j] = B[i*dim1+j] / sum;
      //printf("B[%d][%d]: %f\n", i,j,  B[i*dim1+j]);
    }
  }

}

extern "C" {

void multitensor_op(
        const float *tensorA, // Output Tensor
        float *tensorB,
        int dim0,      
        int dim1,  
        int kernel_id
        )
{

#pragma HLS INTERFACE m_axi port=tensorA offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=tensorB offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=tensorA  bundle=control
#pragma HLS INTERFACE s_axilite port=tensorB  bundle=control
#pragma HLS INTERFACE s_axilite port=dim0 bundle=control
#pragma HLS INTERFACE s_axilite port=dim1 bundle=control
#pragma HLS INTERFACE s_axilite port=kernel_id bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
   switch (kernel_id) {
       case 9: relu(tensorA, tensorB, dim0); break;
       case 10: softmax(tensorA, tensorB, dim0, dim1);break;
   }

}


} 
