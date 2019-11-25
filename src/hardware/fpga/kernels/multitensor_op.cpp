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

   //A[dim1][dim2] column-wise 
   for (int i=0; i<dim1; i++){ 
      // get the max
      max = A[i]; 
      for (int j=0; j< dim0; j++){
         if (A[j*dim1 +i] > max)
            max= A[j*dim1+i];        
      }   
      // Get the sum
      sum = 0;
      for (int j=0; j < dim0; j++) {
         B[j*dim1+i] = exp(A[j*dim1+i]-max);
         sum = sum + B[j*dim1+i];
      }

      // Get the probabilities
      for (int j=0; j < dim0; j++) {
         B[j*dim1+i] = B[j*dim1+i] / sum;
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
